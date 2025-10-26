"""
fema_p695_module.py

FEMA P-695 Spectral Shape Factor (SSF) and Mean Annual Frequency (MAF)
analysis toolkit.

Implements the following main classes:
    - SacIndexRelationship
    - SSF
    - HazardCurve
    - MAF
    - FEMAP695SSF  (primary user-facing class)

Author: Zhiyang Mao, Christianos Burlotos(Stanford University)
Email: zymao@stanford.edu
Created: Oct 2025
Version: 2.0
License: MIT License

Description:
------------
This module provides a fully automated workflow for computing FEMA P-695
spectral shape factors and collapse fragility metrics based on ground motion
records, hazard data, and regression relationships.
"""
__author__ = "Zhiyang Mao, Christianos Burlotos"

__email__ = "zymao@stanford.edu"

__version__ = "2.0"

__license__ = "MIT"

__all__ = ["FEMAP695SSF"]
# import necessary packages
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import requests
from pygmm.model import Scenario
import pygmm.boore_stewart_seyhan_atkinson_2014 as bssa
from scipy.interpolate import interp1d
from scipy.stats import gmean
import statsmodels.api as sm
from scipy.stats import norm

# class for target
class TargetSpectrum:
    """
    Compute target spectra and related quantities (UHS, CMS, target epsilon, Sa ratio)
    for a given site and structure period using USGS deaggregation and BSSA2014 GMM.

    Attributes:
        T (float): Fundamental period of the structure (s).
        longitude (float): Site longitude in degrees (W is negative). Must be within [-125, -65] for USGS APIs.
        latitude (float): Site latitude in degrees. Must be within [24.4, 50] for USGS APIs.
        Vs30 (float): Time-averaged shear-wave velocity in the top 30 m (m/s).
        return_period (int | float): Return period for UHS (years).
        region (str): Region keyword for the GMM distance attenuation/basin model.
            One of {"global", "california", "china", "italy", "japan", "new_zealand", "taiwan", "turkey"}.
        mechanism (str): Faulting mechanism. One of {"U", "SS", "NS", "RS"}.
        Z10 (float): Depth to 1.0 km/s horizon (km). Use -1 if unknown (model will infer).
        a (float): Lower bound factor for period window [a*T, b*T] used in Sa ratio.
        b (float): Upper bound factor for period window [a*T, b*T] used in Sa ratio.
        version (str): USGS deaggregation vintage. One of {"2008", "2023"}.
        Tlist_spectrum (np.ndarray): Period grid for spectra computations; shape (N,), from 0.01 to 10.0 s with 0.01 step.

        # Populated by methods (None until computed):
        UHS_df (pd.DataFrame): UHS and deaggregation summary with columns:
            ["T", f"Sa (RP={return_period})", "M", "R"] where T (s), Sa (g), M (Mw), R (km).
        target_epsilon (float): Epsilon at period T computed by comparing UHS and median GMM.
        CMS (dict): Conditional mean spectrum results with keys:
            "median" -> np.ndarray of Sa medians (g) on Tlist_spectrum;
            "std ln(Sa)" -> np.ndarray of ln standard deviations on Tlist_spectrum.
        target_SaRatio (float): Sa(T)/gmean(Sa over [aT, bT]) based on CMS.
    """
    def __init__(self,T,longitude,latitude,Vs30,return_period,a,b,version,region="california",mechanism="SS",Z10=-1):
        """
        Initialize target-spectrum settings and print a basic info header.

        Args:
            T (float): Structure period (s).
            longitude (float): Site longitude (deg), expected in [-125, -65] for USGS APIs.
            latitude (float): Site latitude (deg), expected in [24.4, 50] for USGS APIs.
            Vs30 (float): Site Vs30 (m/s).
            return_period (int | float): Return period for UHS (years).
            a (float): Lower factor for the Sa ratio period band.
            b (float): Upper factor for the Sa ratio period band.
            version (str): USGS deaggregation version, "2008" or "2023".
            region (str, optional): Region keyword for GMM. Default "california".
            mechanism (str, optional): Faulting mechanism: "U", "SS", "NS", or "RS". Default "SS".
            Z10 (float, optional): Depth to Vs=1.0 km/s (km); -1 to let GMM infer. Default -1.

        Raises:
            ValueError: If `version` is not "2008" or "2023".
        """
        print("="*50)
        print("Start of target calculation section")
        print("-"*50)
        self.T=T
        self.longitude=longitude
        self.latitude=latitude
        self.Vs30=Vs30
        self.return_period=return_period
        self.region=region
        self.mechanism=mechanism
        self.Z10=Z10
        self.a=a
        self.b=b
        self.version=version
        self.Tlist_spectrum=np.arange(0.01,10.01,0.01) # a list of T for CMS
        # check the version of deaggregation
        if version=="2008":
            print("You are using 2008 USGS Deaggregation")
        elif version=="2023":
            print("You are using 2023 USGS Deaggregation")
        else:
            raise ValueError("Only '2008' and '2023' versions are support ed")
        print("-"*50)
        print(" "*16+"Basic Information"+" "*17)
        print("T: ",self.T," s")
        print("Longitude: ",self.longitude)
        print("Latitude: ",self.latitude)
        print("Vs30: ",self.Vs30," m/s")
        print("a: ",self.a)
        print("b: ",self.b)
        print("Return Period: ",self.return_period," years")
    
    def UHSfromUSGS08(self):
        
        """
        Fetch UHS and deaggregation summaries from the USGS 2008 (WUS) API for fixed IMTs.

        Uses per-IMT endpoints for periods: [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 3.0] s.
        Populates `self.UHS_df` with columns ["T", f"Sa (RP={return_period})", "M", "R"].

        Args:
            None

        Returns:
            None

        Side Effects:
            Sets `self.UHS_df` (pd.DataFrame).

        Raises:
            ValueError: If `longitude` or `latitude` are out of accepted bounds.
            RuntimeError: If the USGS API response is not HTTP 200.
        """

        
        # Check the longitude and latitude is in the range that USGS can accept
        if self.longitude <-125 or self.longitude>-65:
            raise ValueError("Longitude out of range(bound: [-125,-65])")
        if self.latitude <24.4 or self.latitude>50:
            raise ValueError("LLatitude out of range(bound: [24.4,50])")
        print("-"*50)
        print("start running API to get USGS Deaggregation.")
        Sa_List=[]
        M_List=[]
        R_List=[]
        periods=[0.1,0.2,0.3,0.5,0.75,1.0,2.0,3.0]
        imts=["SA0P1","SA0P2","SA0P3","SA0P5","SA0P75","SA1P0","SA2P0","SA3P0"]
        for i in range(len(periods)):
            url=f"https://earthquake.usgs.gov/nshmp-haz-ws/deagg/E2008/WUS/{self.longitude}/{self.latitude}/{imts[i]}/{self.Vs30}/{self.return_period}"
            response=requests.get(url)
            # check the response status
            if response.status_code == 200:
                data=response.json()
            else:
                raise RuntimeError("Fail to get the response from USGS Hazard Toolbox")
            # write in the Sa, M and R
            Sa_List.append(data["response"][0]["data"][0]["summary"][0]["data"][2]["value"])
            M_List.append(data["response"][0]["data"][0]["summary"][3]["data"][0]["value"])
            R_List.append(data["response"][0]["data"][0]["summary"][3]["data"][1]["value"])
        # Put all teh result in a DF
        output_dict={"T":periods,"Sa (RP={})".format(self.return_period):Sa_List,"M":M_List,"R":R_List}
        output=pd.DataFrame(output_dict)
        self.UHS_df=output
        print("UHS Calculation Done.")   
        return
    def UHSfromUSGS23(self):
        """
        Fetch UHS and deaggregation summaries from the USGS 2023 CONUS API (dynamic/disagg).

        Queries multiple IMTs in one request across periods:
        [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
         0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0] s.
        Populates `self.UHS_df` with columns ["T", f"Sa (RP={return_period})", "M", "R"].

        Args:
            None

        Returns:
            None

        Side Effects:
            Sets `self.UHS_df` (pd.DataFrame).

        Raises:
            ValueError: If `longitude` or `latitude` are out of accepted bounds.
            RuntimeError: If the USGS API response is not HTTP 200.
        """
        # Check the longitude and latitude is in the range that USGS can accept
        if self.longitude <-125 or self.longitude>-65:
            raise ValueError("Longitude out of range(bound: [-125,-65])")
        if self.latitude <24.4 or self.latitude>50:
            raise ValueError("LLatitude out of range(bound: [24.4,50])")
        print("-"*50)
        print("start running API to get USGS Deaggregation")
        # create the url for the request (can cause problem for older version of python, in that case, switch to .format())
        url=f"https://earthquake.usgs.gov/ws/nshmp/conus-2023/dynamic/disagg/{self.longitude}/{self.latitude}/{self.Vs30}/{self.return_period}?imt=SA0P01&imt=SA0P02&imt=SA0P03&imt=SA0P05&imt=SA0P075&imt=SA0P1&imt=SA0P15&imt=SA0P2&imt=SA0P25&imt=SA0P3&imt=SA0P4&imt=SA0P5&imt=SA0P75&imt=SA1P0&imt=SA1P5&imt=SA2P0&imt=SA3P0&imt=SA4P0&imt=SA5P0&imt=SA7P5&imt=SA10P0"

        # request for the data
        response=requests.get(url)

        # check the response status
        if response.status_code == 200:
            data=response.json()
        else:
            raise RuntimeError("Fail to get the response from USGS Hazard Toolbox")

        # Build the period list (from the USGS Hazard Tool)
        Period=[0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1.0,1.5,2.0,3.0,4.0,5.0,7.5,10.0]

        # Get the Sa, M and R from the response
        Sa_List=[]
        M_List=[]
        R_List=[]
        for i in range(len(Period)):
            Sa_List.append(data['response']["disaggs"][i]["data"][0]["summary"][0]['data'][2]["value"])
            M_List.append(data['response']["disaggs"][i]["data"][0]['summary'][3]['data'][0]['value'])
            R_List.append(data['response']["disaggs"][i]["data"][0]['summary'][3]['data'][1]['value'])

        # Put all the reulst in a DataFrame
        output_dict = {"T":Period,"Sa (RP={})".format(self.return_period):Sa_List,"M":M_List,"R":R_List}
        output=pd.DataFrame(output_dict)
        self.UHS_df=output
        print("UHS Calculation Done.")
        return
    def BSSA2014(self,T, M, Rjb, Vs30, mechanism, region, Z10):
        """
        Evaluate BSSA2014 GMM for a given scenario and period.

        Args:
            T (float): Oscillator period (s).
            M (float): Moment magnitude (Mw).
            Rjb (float): Joyner-Boore distance to rupture (km).
            Vs30 (float): Site Vs30 (m/s).
            mechanism (str): Faulting mechanism, one of {"U", "SS", "NS", "RS"}.
            region (str): Region keyword for GMM ("global", "california", ...).
            Z10 (float): Depth to Vs=1.0 km/s (km). Use -1 if unknown.

        Returns:
            tuple[float, float]:
                - median_Sa (float): Median spectral acceleration at period T (g).
                - ln_stdev_Sa (float): Natural log of the standard deviation of Sa(T).

        Notes:
            If `Z10 == -1`, the scenario omits depth_1_0 and lets the model infer/ignore it.
        """

        # Construct the earthquake scenario
        if Z10 != -1:
            earthquake_scenario = Scenario(mag=M, dist_jb=Rjb, v_s30=Vs30, mechanism=mechanism, region=region, depth_1_0=Z10)
        else:
            earthquake_scenario = Scenario(mag=M, dist_jb=Rjb, v_s30=Vs30, mechanism=mechanism, region=region)

        # Initialize the BSSA2014 model with the scenario
        gmm = bssa.BooreStewartSeyhanAtkinson2014(scenario = earthquake_scenario)

        # Get the median spectral acceleration at the specified period
        median_Sa = gmm.interp_spec_accels(T)

        # Get the natural logarithm of the standard deviation of spectral acceleration
        ln_stdev_Sa = gmm.interp_ln_stds(T, kind='linear')

        return median_Sa, ln_stdev_Sa
    def compute_targete_epsilon(self):
        """
        Compute target epsilon at period `self.T` by comparing UHS and BSSA2014 median Sa.

        Workflow:
            1) Interpolate UHS Sa(T) from `self.UHS_df`.
            2) Obtain (M, R) at/around T and compute median and ln-std from BSSA2014.
            3) Compute epsilon = [ln(Sa_UHS) - ln(median_Sa_gmm)] / ln_stdev_Sa_gmm.

        Args:
            None

        Returns:
            None

        Side Effects:
            Sets `self.target_epsilon` (float).

        Requires:
            `self.UHS_df` must be populated by `UHSfromUSGS08()` or `UHSfromUSGS23()`.

        Raises:
            KeyError/IndexError: If `self.UHS_df` is missing required columns.
            ValueError: If interpolation bounds are invalid for provided period.
        """
        # Interpolate Sa from the UHS at the building period [THIS DOESN'T FOLLOW PATTERN BUT HAS TO BE DONE?]
        print("-"*50)
        print(" "*18,"Target Epsilon"," "*18)
        Sa_UHS = np.interp(np.log(self.T), np.log(self.UHS_df['T']), self.UHS_df.iloc[:,1])
        print("Mean Sa(T): ",Sa_UHS," g")
        # Check if T1 is exactly in the 'T' column
        if self.T in self.UHS_df['T'].values:
            exact_row = self.UHS_df[self.UHS_df['T'] == self.T].iloc[0]
            M = exact_row['M']
            Rjb = exact_row['R']
            median_Sa_gmm, ln_stdev_Sa_gmm = self.BSSA2014(self.T, M, Rjb, self.Vs30, self.mechanism, self.region, self.Z10)
            print("Mean M: ",M)
            print("Mean R: ",Rjb," km")
        else:
            # Get the rows just below and just above T
            below_rows = self.UHS_df[self.UHS_df['T'] < self.T]
            above_rows = self.UHS_df[self.UHS_df['T'] > self.T]
            if below_rows.empty or above_rows.empty:
                print("T1 is out of bounds of the T values in the DataFrame.")
            else:
                below = below_rows.iloc[-1]
                above = above_rows.iloc[0]

                # Extract M and R at those rows
                M_below, Rjb_below = below['M'], below['R']
                M_above, Rjb_above = above['M'], above['R']

                # Call GMM to calculate median and stdev of Sa(T) at both above and below locations
                median_Sa_gmm_below, ln_stdev_Sa_gmm_below = self.BSSA2014(self.T, M_below, Rjb_below, self.Vs30, self.mechanism, self.region, self.Z10)
                median_Sa_gmm_above, ln_stdev_Sa_gmm_above = self.BSSA2014(self.T, M_above, Rjb_above, self.Vs30, self.mechanism, self.region, self.Z10)

                # Interpolate after running GMM
                both_periods = np.array([below['T'], above['T']])
                both_median = np.array([median_Sa_gmm_below, median_Sa_gmm_above])
                both_stdev = np.array([ln_stdev_Sa_gmm_below, ln_stdev_Sa_gmm_above])

                median_Sa_gmm = np.interp(np.log(self.T), np.log(both_periods), both_median)
                ln_stdev_Sa_gmm = np.interp(np.log(self.T), np.log(both_periods), both_stdev)
                print("Mean M ", np.interp(np.log(self.T),np.log([below["T"],above["T"]]),[M_below,M_above]))
                print("Mean R: ",np.interp(np.log(self.T),np.log([below["T"],above["T"]]),[Rjb_below,Rjb_above]), " km")
        # Compute target epsilon
        target_epsilon = (np.log(Sa_UHS) - np.log(median_Sa_gmm)) / ln_stdev_Sa_gmm
        print("Target epsilon: ",target_epsilon)
        print("Finish calculating target epsilon.")
        self.target_epsilon=target_epsilon
        return
    def compute_rho_BakerJayaram2008(self):
        """
        Compute period-to-period correlation rho(T, T*) per Baker & Jayaram (2008).

        Uses `self.Tlist_spectrum` as T-grid and `self.T` as conditioning period T*.

        Args:
            None

        Returns:
            np.ndarray: Correlation coefficients rho with shape (len(self.Tlist_spectrum),).

        Notes:
            Implements piecewise closed forms (C1-C4) from Baker & Jayaram (2008).
        """

        # Initialize output vector
        rho = np.zeros(len(self.Tlist_spectrum))

        for idx, period in enumerate(self.Tlist_spectrum):

            # Determine Tmin and Tmax
            Tmin = min(period, self.T)
            Tmax = max(period, self.T)

            # Calculate coefficient C1
            C1 = 1 - np.cos(np.pi / 2 - 0.366 * np.log(Tmax / max(Tmin, 0.109)))

            # Calculate coefficient C2
            if Tmax < 0.2:
                C2 = 1 - 0.105 * (1 - 1 / (1 + np.exp(100 * Tmax - 5))) * ((Tmax - Tmin) / (Tmax - 0.0099))
            else:
                C2 = 0

            # Calculate coefficient C3
            if Tmax < 0.109:
                C3 = C2
            else:
                C3 = C1

            # Calculate coefficient C4
            C4 = C1 + 0.5 * (np.sqrt(C3) - C3) * (1 + np.cos(np.pi * Tmin / 0.109))

            # Calculate rho
            if Tmax < 0.109:
                rho[idx] = C2
            elif Tmin > 0.109:
                rho[idx] = C1
            elif Tmax < 0.2:
                rho[idx] = min(C2, C4)
            else:
                rho[idx] = C4

        return rho       
    def compute_conditional_mean_spectrum(self):
        """
        Compute the Conditional Mean Spectrum (CMS) on `self.Tlist_spectrum`.

        Workflow:
            1) Compute rho(T, self.T) via Baker–Jayaram (2008).
            2) Interpolate deaggregation means M, R at `self.T` from `self.UHS_df`.
            3) Evaluate BSSA2014 over `self.Tlist_spectrum` to get median and ln-std.
            4) Condition with epsilon: ln Sa_CMS = ln Sa_med + ln_std * rho * target_epsilon.

        Args:
            None

        Returns:
            None

        Side Effects:
            Sets `self.CMS` dict with:
                - "median": np.ndarray of median CMS Sa (g) over `self.Tlist_spectrum`.
                - "std ln(Sa)": np.ndarray of conditional ln standard deviations.

        Requires:
            `self.UHS_df` and `self.target_epsilon` must be set.

        Raises:
            KeyError/IndexError: If `self.UHS_df` lacks required columns.
        """
        print("-"*50)
        print(" "*23,"CMS"," "*24)
        # Compute rho
        rho = self.compute_rho_BakerJayaram2008()

        # Interpolate GMM inputs from the UHS at the building period
        M = np.interp(np.log(self.T), np.log(self.UHS_df['T']), self.UHS_df['M'])
        Rjb = np.interp(np.log(self.T), np.log(self.UHS_df['T']), self.UHS_df['R'])
        print("Mean M: ",M)
        print("Mean R: ", Rjb)

        # Compute Sa spectrum using the GMM over the period range Tlist
        median_Sa_gmm = np.zeros(len(self.Tlist_spectrum))
        ln_stdev_Sa_gmm = np.zeros(len(self.Tlist_spectrum))
        for idx, period in enumerate(self.Tlist_spectrum):
            median_Sa_gmm[idx], ln_stdev_Sa_gmm[idx] = self.BSSA2014(period, M, Rjb, self.Vs30, self.mechanism, self.region, self.Z10)

        # "Condition" the spectrum with rho and target epsilon
        median_Sa_CMS = np.exp(np.log(median_Sa_gmm) + ln_stdev_Sa_gmm * rho * self.target_epsilon)
        ln_stdev_Sa_CMS = ln_stdev_Sa_gmm * np.sqrt(1 - rho**2)
        self.CMS={"median":median_Sa_CMS,"std ln(Sa)":ln_stdev_Sa_CMS}
        print("Finish calculating CMS.")
        return
    def calculate_target_SaRatio(self):
        """
        Compute the target Sa ratio = Sa(T) / gmean(Sa(t), t in [aT, bT]) based on CMS.

        The CMS arrays must exist in `self.CMS`. Period band [aT, bT] is taken from
        `self.Tlist_spectrum` via boolean masking, with linear interpolation to
        include exact endpoints if missing.

        Args:
            None

        Returns:
            None

        Side Effects:
            Sets `self.target_SaRatio` (float).

        Requires:
            `self.CMS` must be populated by `compute_conditional_mean_spectrum()`.

        Raises:
            ValueError: If the [aT, bT] window is invalid or outside the T grid.
        """
        print("-"*50)
        print(" "*21,"Sa Ratio"," "*21)
        # Slice conditional mean spectra from Ta to Tb
        mask = (self.Tlist_spectrum >= self.T*self.a) & (self.Tlist_spectrum <= self.T*self.b)
        periods_range = self.Tlist_spectrum[mask]
        Sa_CMS_range = self.CMS["median"][mask]

        # Add in Ta if its not the first period listed in periods_range
        if periods_range[0] != self.T*self.a:
            Sa_CMS_Ta = np.interp(self.T*self.a, self.Tlist_spectrum, self.CMS["median"])
            Sa_CMS_Ta = np.array([Sa_CMS_Ta])
            Sa_CMS_range = np.concatenate((Sa_CMS_Ta, Sa_CMS_range))

        # Add in Tb if its not the last period listed in periods_range
        if periods_range[-1] != self.T*self.b:
            Sa_CMS_Tb = np.interp(self.T*self.b, self.Tlist_spectrum, self.CMS["median"])
            Sa_CMS_Tb = np.array([Sa_CMS_Tb])
            Sa_CMS_range = np.concatenate((Sa_CMS_range, Sa_CMS_Tb))

        # Compute Sa at building period T using conditional mean spectra
        Sa_CMS_T = np.interp(self.T, self.Tlist_spectrum, self.CMS["median"])
        # Compute target SaRatio
        target_SaRatio = Sa_CMS_T / gmean(Sa_CMS_range)
        self.target_SaRatio=target_SaRatio
        print("Ta: ",self.T*self.a)
        print("Sa(Ta): ",Sa_CMS_Ta[0]," g")
        print("Tb: ",self.T*self.b)
        print("Sa(Ta): ",Sa_CMS_Tb[0]," g")
        print("Target Sa Ratio: ",self.target_SaRatio)
        print("Finish calculating target Sa Ratio.")
        return
    def run_target(self):
        """
        End-to-end pipeline to compute UHS → epsilon → CMS → Sa ratio.

        Steps:
            1) Deaggregation + UHS via `UHSfromUSGS08()` or `UHSfromUSGS23()` based on `self.version`.
            2) Compute `self.target_epsilon`.
            3) Compute CMS (`self.CMS`).
            4) Compute target Sa ratio (`self.target_SaRatio`).

        Args:
            None

        Returns:
            None

        Side Effects:
            Populates `self.UHS_df`, `self.target_epsilon`, `self.CMS`, `self.target_SaRatio`.

        Raises:
            ValueError: If `version` is unsupported.
            RuntimeError: If remote API calls fail.
        """
        # Run deaggregation
        if self.version=="2008":
            self.UHSfromUSGS08()
        elif self.version=="2023":
            self.UHSfromUSGS23()
        else: raise ValueError("Only '2008' and '2023' versions are supported") 
        # compute target epsilon
        self.compute_targete_epsilon()
        # compute CMS
        self.compute_conditional_mean_spectrum()
        # compute target Sa Ratio
        self.calculate_target_SaRatio()
        print("-"*50)
        print("End of target calculation section")
        print("="*50)
        
# Class for Sac-epsilon/SaRatio relationship
class SacIndexRelationship:
    """
    Analyze relationships between collapse capacity (Sac) and either epsilon(T) or SaRatio.

    This class consumes:
      - Record-wise response spectra (self.Sa_Form, columns: 'T' then one column per EQ_ID),
      - Earthquake/site metadata table (self.EQ_info, auto-loaded),
      - A target period window [a*T, b*T].

    It computes per-record epsilon at period T using BSSA2014, per-record SaRatio in [aT, bT],
    and performs simple (log/linear) regressions Sac ~ epsilon and Sac ~ SaRatio, with plots.

    Attributes:
        T (float): Fundamental period of the structure (s).
        a (float): Lower factor; the window lower bound is T_a = a * T.
        b (float): Upper factor; the window upper bound is T_b = b * T.
        Sac_Form (pd.DataFrame): Collapse capacities table with two columns:
            [EQ_ID (int), Sac (float)]. Provided by user to __init__.
        Z10 (float): Depth to Vs=1.0 km/s (km). Fixed as -1 (unknown) for all records.
        record_set (str): One of {"Far Field","Near Field","Combined"}, inferred from Sac_Form row count.
        Sa_Form (pd.DataFrame): Response spectra for records; loaded from
            "./Sa Form/Sa_FF.xlsx", "./Sa Form/Sa_NF.xlsx", or "./Sa Form/Sa_Combined.xlsx".
            Format: first column 'T' (float, s); subsequent columns named by EQ_ID (int), values Sa (g).
        EQ_info (pd.DataFrame): Earthquake/site info loaded from "./EQ info/EQ_info.xlsx".
            Must include columns: ['EQ_ID','M','Rjb','mechanism','Vs30','region'].
        epsilon (pd.DataFrame): Computed per-record epsilon at T.
            Columns: ['EQ_ID' (int), 'epsilon' (float)]. Set by epsilon_record().
        SaRatio (pd.DataFrame): Computed per-record SaRatio.
            Columns: ['EQ_ID' (int), 'Sa Ratio' (float)]. Set by SaRatio_record().
        epislon_regression_result (dict):  # Note: key name retains original spelling
            Results of Sac ~ epsilon regression. Keys:
                - "Original Data": {"EQ_ID": list[int], "Sac": list[float], "Epsilon": list[float]}
                - "Plot Data": {"Sac": list[float], "Epsilon": list[float]}  # fitted curve
                - "beta": list[float] (slope, intercept) from OLS on ln(Sac) = b0*epsilon + b1
                - "Correlation": float, corr(epsilon, ln(Sac))
        SaRatio_regression_result (dict):
            Results of Sac ~ SaRatio regression. Keys:
                - "Original Data": {"EQ_ID": list[int], "Sac": list[float], "Sa Ratio": list[float]}
                - "Plot Data": {"Sac": list[float], "Sa Ratio": list[float]}  # fitted line
                - "beta": list[float] (slope, intercept) from OLS on Sac = b0*SaRatio + b1
                - "Correlation": float, corr(SaRatio, Sac)
    """
    def __init__(self,T,a,b,Sac_Form):
        """
        Initialize analysis settings and load auxiliary data/spectra.

        Infers which record set is used by the number of rows in `Sac_Form`:
            44 -> "Far Field", 56 -> "Near Field", 100 -> "Combined".
        Loads corresponding spectra file into `self.Sa_Form` and EQ metadata into `self.EQ_info`.

        Args:
            T (float): Fundamental period (s).
            a (float): Lower factor for period window (T_a = a*T).
            b (float): Upper factor for period window (T_b = b*T).
            Sac_Form (pd.DataFrame): Two-column DataFrame with:
                - Column 0: EQ_ID (int)
                - Column 1: Sac collapse capacity (float)

        Raises:
            ValueError: If the row count of Sac_Form does not match any supported record set.

        Side Effects:
            Prints a short header; loads Excel files from "./Sa Form/" and "./EQ info/".
        """
        print("="*50)
        print("Start of Sac-epsilon/SaRatio relationship section")
        self.Sac_Form=Sac_Form
        self.T=T
        self.a=a
        self.b=b
        self.Z10=-1 #No available data for all the z10,so -1 is used for all the record
        # check the record set and read corresponding the Sa Form
        if self.Sac_Form.shape[0]==44: 
            self.record_set="Far Field"
            self.Sa_Form=pd.read_excel("./Sa Form/Sa_FF.xlsx")
        elif self.Sac_Form.shape[0]==56:
            self.record_set="Near Field"
            self.Sa_Form=pd.read_excel("./Sa Form/Sa_NF.xlsx")
        elif self.Sac_Form.shape[0]==100:
            self.record_set="Combined"
            self.Sa_Form=pd.read_excel("./Sa Form/Sa_Combined.xlsx")
        else:
            raise ValueError(f"The Sac Form is not belonged to any of the ground motion set!\nexpected: 44 rows for Far Field, 56 rows for Near Field, 100 rows for Combined\nbut get: {self.Sac_Form.shape[0]}")
        self.EQ_info=pd.read_excel('./EQ info/EQ_info.xlsx')
        print("-"*50)
        print(" "*16+"Basic Information"+" "*17)
        print("T: ",self.T," s")
        print("a: ",self.a)
        print("b: ",self.b)
        print("Record Set: ",self.record_set)
    def BSSA2014(self,T, M, Rjb, Vs30, mechanism, region, Z10):
        """
        Evaluate BSSA2014 GMM for a given record/site scenario and period.

        Args:
            T (float): Oscillator period (s).
            M (float): Moment magnitude (Mw).
            Rjb (float): Joyner-Boore distance to rupture (km).
            Vs30 (float): Site Vs30 (m/s).
            mechanism (str): Faulting mechanism, one of {"U", "SS", "NS", "RS"}.
            region (str): GMM region keyword ("global","california","china","italy",
                "japan","new_zealand","taiwan","turkey").
            Z10 (float): Depth to Vs=1.0 km/s (km). Use -1 to omit/infer.

        Returns:
            tuple[float, float]:
                - median_Sa (float): Median spectral acceleration at T (g).
                - ln_stdev_Sa (float): Natural log of the standard deviation at T.

        Notes:
            If `Z10 == -1`, the scenario omits depth_1_0 in the model initialization.
        """
        # Construct the earthquake scenario
        if Z10 != -1:
            earthquake_scenario = Scenario(mag=M, dist_jb=Rjb, v_s30=Vs30, mechanism=mechanism, region=region, depth_1_0=Z10)
        else:
            earthquake_scenario = Scenario(mag=M, dist_jb=Rjb, v_s30=Vs30, mechanism=mechanism, region=region)

        # Initialize the BSSA2014 model with the scenario
        gmm = bssa.BooreStewartSeyhanAtkinson2014(scenario = earthquake_scenario)

        # Get the median spectral acceleration at the specified period
        median_Sa = gmm.interp_spec_accels(T)

        # Get the natural logarithm of the standard deviation of spectral acceleration
        ln_stdev_Sa = gmm.interp_ln_stds(T, kind='linear')

        return median_Sa, ln_stdev_Sa
    def epsilon_record(self):
        """
        Compute epsilon(T) for each record: epsilon = [ln(Sa_rec(T)) − ln(Sa_med_GMM)] / ln_std_GMM.

        Uses BSSA2014 with per-record (M, Rjb, Vs30, mechanism, region) from `self.EQ_info`,
        and Sa_rec(T) interpolated from `self.Sa_Form`.

        Args:
            None

        Returns:
            None

        Side Effects:
            Sets `self.epsilon` (pd.DataFrame) with columns ['EQ_ID','epsilon'].

        Requires:
            - self.Sa_Form with columns: 'T' (float s), then EQ_ID columns (int) containing Sa (g).
            - self.EQ_info with columns: ['EQ_ID','M','Rjb','mechanism','Vs30','region'].
            - self.record_set correctly set by __init__.
        """
        # Import 5% damped response spectra for each record in the specified RecordSet
        if self.record_set == 'Far Field':
            records_list =self.EQ_info['EQ_ID'][:44].to_list()
        if self.record_set == 'Near Field':
            records_list =self.EQ_info['EQ_ID'][44:].to_list()
        if self.record_set == 'Combined':
            records_list =self.EQ_info['EQ_ID'].to_list()

        # Define periods associated with the record spectra
        periods = self.Sa_Form['T']

        # Compute spectral acceleration at T for both target spectra and record spectra
        median_Sa_gmm = np.zeros(len(records_list))
        ln_stdev_Sa_gmm = np.zeros(len(records_list))
        Sa_records = np.zeros(len(records_list))

        for idx,record in enumerate(records_list):

            # Compute Sa_GMM (SaT1 of target spectra) for each record site via GMM
            M = self.EQ_info.loc[self.EQ_info['EQ_ID'] == record, 'M'].values[0]
            Rjb = self.EQ_info.loc[self.EQ_info['EQ_ID'] == record, 'Rjb'].values[0]
            mechanism = self.EQ_info.loc[self.EQ_info['EQ_ID'] == record, 'mechanism'].values[0]
            Vs30 = self.EQ_info.loc[self.EQ_info['EQ_ID'] == record, 'Vs30'].values[0]
            region = self.EQ_info.loc[self.EQ_info['EQ_ID'] == record, 'region'].values[0]

            median_Sa_gmm[idx], ln_stdev_Sa_gmm[idx] = self.BSSA2014(self.T, M, Rjb, Vs30, mechanism, region, self.Z10)

            # Compute Sa_record (SaT1 of record spectra) for each record in the set
            record_spectra = self.Sa_Form[record]
            Sa_records[idx] = np.interp(self.T, periods, record_spectra)

        # Compute epsilon for each record in the set
        epsilon = (np.log(Sa_records) - np.log(median_Sa_gmm)) / ln_stdev_Sa_gmm

        # Prepare output dataframe with EQ_ID and epsilon
        epsilon={"EQ_ID":records_list,"epsilon":epsilon}
        epsilon=pd.DataFrame(epsilon)
        self.epsilon=epsilon
        return
    def SaRatio_record(self):
        """
        Compute SaRatio per record: Sa(T) / gmean(Sa(t), t in [aT, bT]) from response spectra.

        The spectra table must have first column 'T' (s), and subsequent columns named by EQ_ID (int).
        For each record (column), Sa(T), Sa(T_a), Sa(T_b) are obtained by linear interpolation.

        Args:
            None

        Returns:
            None

        Side Effects:
            Sets `self.SaRatio` (pd.DataFrame) with columns ['EQ_ID','Sa Ratio'] sorted by EQ_ID.

        Raises:
            ValueError:
                - If first column is not named 'T'.
                - If any non-first column name is not an int EQ_ID.
                - If any column dtype is not float.
        """
        # check the fromat of the DataFrame
        Column_Names=self.Sa_Form.columns
        # check the first column is T
        if Column_Names[0] != "T":
            raise ValueError("The first columne should be the period, and please name the first column 'T'")
            return 0
        # check other columns are Sa for each gm record
        for i in range(1,self.Sa_Form.shape[1]):
            if type(Column_Names[i]) != int:
                raise ValueError("The column {} should be the Sa for the ground motion record, and please name the column using EQ_ID".format(i+1))
                return 0
        # check the value in the spreadsheet
        Column_Types=self.Sa_Form.dtypes
        for i in range(len(Column_Types)):
            if Column_Types.iloc[i]!=float:
                raise ValueError("Column {} is not the float number".format(i+1))

        # build the fist col for the output dataframe
        record_name=self.Sa_Form.columns[1:]
        output_dict={"EQ_ID":record_name}

        # Define Tmin and Tmax
        T_a=self.a*self.T
        T_b=self.b*self.T

        # Loop to calculate the SA Ratio for each record in the set
        Sa_Ratio=[]
        for i in range(1,self.Sa_Form.shape[1]):
            # Find the SA at T
            for j in range(self.Sa_Form.shape[0]):
                if self.Sa_Form.iloc[j,0]>self.T:
                    Sa_T=np.interp(self.T,self.Sa_Form.iloc[j-1:j+1,0],self.Sa_Form.iloc[j-1:j+1,i])
                    break
            # Find the SA at [T_a:0.01:T_b]
            Sa_average_list=[]
            # Find Sa for Ta in the form
            for j in range(self.Sa_Form.shape[0]):
                if self.Sa_Form.iloc[j,0]>T_a:
                    Sa_Ta=np.interp(T_a,self.Sa_Form.iloc[j-1:j+1,0],self.Sa_Form.iloc[j-1:j+1,i])
                    Ta_index_start=j
                    break
            # Find Sa for Tb in the form
            for j in range(self.Sa_Form.shape[0]):
                if self.Sa_Form.iloc[j,0]>T_b:
                    Sa_Tb=np.interp(T_b,self.Sa_Form.iloc[j-1:j+1,0],self.Sa_Form.iloc[j-1:j+1,i])
                    Tb_index_end=j-1
                    break
            Sa_average_list.append(Sa_Ta)
            Sa_average_mid=self.Sa_Form.iloc[Ta_index_start:Tb_index_end+1,i]
            Sa_average_mid.tolist()
            Sa_average_list.extend(Sa_average_mid)
            Sa_average_list.append(Sa_Tb)
            # Calculate the geomean to get the Sa_average
            Sa_avg=gmean(Sa_average_list)
            # Calculate the SA Ratio
            Sa_Ratio.append(Sa_T/Sa_avg)
        output_dict["Sa Ratio"]=Sa_Ratio
        output=pd.DataFrame(output_dict)
        output=output.sort_values(by="EQ_ID")
        self.SaRatio=output
        return
    def regression_epsilon_Sac(self):
        """
        Run log-linear regression: ln(Sac) = β0 * epsilon + β1; compute correlation.

        Args:
            None

        Returns:
            None

        Side Effects:
            Sets `self.epislon_regression_result` (dict) containing original data, fitted curve,
            regression coefficients β = [β0, β1], and correlation corr(epsilon, ln(Sac)).

        Requires:
            - self.Sac_Form with columns [EQ_ID, Sac].
            - self.epsilon with columns [EQ_ID, epsilon].

        Raises:
            ValueError: If EQ_ID ordering/membership does not match between the two tables.
        """
        # rearange the two DataFrame to make sure the EQ_ID are in the same order
        Sac_Form=self.Sac_Form.sort_values(by=self.Sac_Form.columns[0])
        Epsilon_Form=self.epsilon.sort_values(by=self.epsilon.columns[0])
        # check the name of the records corresponds in the two DataFrame
        for i in range(Sac_Form.shape[0]):
            if Sac_Form.iloc[i,0]!=Epsilon_Form.iloc[i,0]:
                raise ValueError("The Ground Motion Records in Sa Collapse and Epsilon are not the same")
        # Calculate the relationship between Sa collapse and Epsilon
        X=np.ones([Epsilon_Form.shape[0],2])
        # Loop to fill in the Epsilon
        for i in range(Epsilon_Form.shape[0]):
            X[i][0]=Epsilon_Form.iloc[i,1]
        Y=np.log(Sac_Form.iloc[:,1].tolist())
        beta=np.linalg.inv(X.T@X)@X.T@Y
        x_plot=np.arange(np.min(Epsilon_Form.iloc[:,1]),np.max(Epsilon_Form.iloc[:,1]),0.01)
        y_plot=np.exp(x_plot*beta[0]+beta[1])
        cof=np.corrcoef(Epsilon_Form.iloc[:,1],Y)

        # Build the output dict
        # The Original Data
        Original_Data={"EQ_ID":Sac_Form.iloc[:,0].tolist(),
                    "Sac":Sac_Form.iloc[:,1].tolist(),
                    "Epsilon":Epsilon_Form.iloc[:,1].tolist()}
        # The fitting data using the regression result
        Plot_Data={"Sac":y_plot.tolist(),
                "Epsilon":x_plot.tolist()}
        # Put all the result in one dict
        Output={"Original Data":Original_Data,
            "Plot Data":Plot_Data,
            "beta":beta.tolist(),
            "Correlation":cof[0][1]}
        self.epislon_regression_result=Output
        return
    def plot_epsilon_Sac(self):
        """
        Plot scatter of epsilon vs Sac and overlay the regression curve from `epislon_regression_result`.

        Args:
            None

        Returns:
            None

        Requires:
            `self.epislon_regression_result` produced by `regression_epsilon_Sac()`.

        Side Effects:
            Displays a Matplotlib figure and legend; no object is returned.
        """
        # create the plot
        fig,ax=plt.subplots()

        # Plot the original data in scatter points
        ax.scatter(self.epislon_regression_result["Original Data"]["Epsilon"],self.epislon_regression_result["Original Data"]["Sac"])

        # Plot the Regression Curve
        ax.plot(self.epislon_regression_result["Plot Data"]["Epsilon"],self.epislon_regression_result["Plot Data"]["Sac"],label=r"ln(y)={}x+{}, $\rho$={}".format(round(self.epislon_regression_result["beta"][0],2),round(self.epislon_regression_result["beta"][1],2),round(self.epislon_regression_result["Correlation"],2)))

        ax.set_xlabel(r"$\epsilon(T_1)$")
        ax.set_ylabel(r"$Sa_{collapse}$")
        ax.set_title(r"$Sa_{collapse}$-$\epsilon$ Relationship")
        ax.grid(True)
        ax.legend()
        plt.show()
    def regression_SaRatio_Sac(self):
        """
        Run linear regression: Sac = β0 * SaRatio + β1; compute correlation.

        Args:
            None

        Returns:
            None

        Side Effects:
            Sets `self.SaRatio_regression_result` (dict) containing original data, fitted line,
            regression coefficients β = [β0, β1], and correlation corr(SaRatio, Sac).

        Requires:
            - self.Sac_Form with columns [EQ_ID, Sac].
            - self.SaRatio with columns [EQ_ID, Sa Ratio].

        Raises:
            ValueError: If EQ_ID ordering/membership does not match between the two tables.
        """
        # rearange the two DataFrame to make sure the EQ_ID are in the same order
        Sac_Form=self.Sac_Form.sort_values(by=self.Sac_Form.columns[0])
        SaRatio_Form=self.SaRatio.sort_values(by=self.SaRatio.columns[0])
        # check the name of the records corresponds in the two DataFrame
        for i in range(Sac_Form.shape[0]):
            if Sac_Form.iloc[i,0]!=SaRatio_Form.iloc[i,0]:
                raise ValueError("The Ground Motion Records in Sa Collapse and Sa Ratio are not the same")
        # Calculate the relationship between Sa collapse and Epsilon
        X=np.ones([SaRatio_Form.shape[0],2])
        # Loop to fill in the Epsilon
        for i in range(SaRatio_Form.shape[0]):
            X[i][0]=SaRatio_Form.iloc[i,1]
        Y=Sac_Form.iloc[:,1].tolist()
        beta=np.linalg.inv(X.T@X)@X.T@Y
        x_plot=np.arange(np.min(SaRatio_Form.iloc[:,1]),np.max(SaRatio_Form.iloc[:,1]),0.01)
        y_plot=x_plot*beta[0]+beta[1]
        cof=np.corrcoef(SaRatio_Form.iloc[:,1],Y)

        # Build the output dict
        # The Original Data
        Original_Data={"EQ_ID":Sac_Form.iloc[:,0].tolist(),
                    "Sac":Sac_Form.iloc[:,1].tolist(),
                    "Sa Ratio":SaRatio_Form.iloc[:,1].tolist()}
        # The fitting data using the regression result
        Plot_Data={"Sac":y_plot.tolist(),
                "Sa Ratio":x_plot.tolist()}
        # Put all the result in one dict
        Output={"Original Data":Original_Data,
            "Plot Data":Plot_Data,
            "beta":beta.tolist(),
            "Correlation":cof[0][1]}
        self.SaRatio_regression_result=Output
        return
    def plot_SaRatio_Sac(self):
        """
        Plot scatter of SaRatio vs Sac and overlay the regression line from `SaRatio_regression_result`.

        Args:
            None

        Returns:
            None

        Requires:
            `self.SaRatio_regression_result` produced by `regression_SaRatio_Sac()`.

        Side Effects:
            Displays a Matplotlib figure and legend; no object is returned.
        """
        # create the plot
        fig,ax=plt.subplots()

        # Plot the original data in scatter points
        ax.scatter(self.SaRatio_regression_result["Original Data"]["Sa Ratio"],self.SaRatio_regression_result["Original Data"]["Sac"])

        # Plot the Regression Curve
        ax.plot(self.SaRatio_regression_result["Plot Data"]["Sa Ratio"],self.SaRatio_regression_result["Plot Data"]["Sac"],label=r"y={}x+{}, $\rho$={}".format(round(self.SaRatio_regression_result["beta"][0],2),round(self.SaRatio_regression_result["beta"][1],2),round(self.SaRatio_regression_result["Correlation"],2)))

        ax.set_xlabel(r"$Sa\ Ratio$")
        ax.set_ylabel(r"$Sa_{collapse}$")
        ax.set_title(r"$Sa_{collapse}$-$Sa\ Ratio$ Relationship")
        ax.grid(True)
        ax.legend()
        plt.show()
    def run_Sac_index(self):
        """
        End-to-end workflow: epsilon(T) per record → SaRatio per record → two regressions + plots.

        Steps:
            1) epsilon_record(): compute per-record epsilon(T).
            2) SaRatio_record(): compute per-record SaRatio in [aT, bT].
            3) regression_epsilon_Sac() + plot_epsilon_Sac().
            4) regression_SaRatio_Sac() + plot_SaRatio_Sac().

        Args:
            None

        Returns:
            None

        Side Effects:
            Populates `self.epsilon`, `self.SaRatio`,
            `self.epislon_regression_result`, `self.SaRatio_regression_result`,
            and displays two plots.
        """
        self.epsilon_record()
        self.SaRatio_record()
        print("-"*50)
        print("Plots:")
        self.regression_epsilon_Sac()
        self.plot_epsilon_Sac()
        self.regression_SaRatio_Sac()
        self.plot_SaRatio_Sac()
        print("-"*50)
        print("End of Sac-epsilon/SaRatio relationship section")
        print("="*50)

# Class for Calculating SSF and shift the fragility curve
class SSF:
    """
    Compute Spectral Shape Factors (SSF) from target spectra and data-driven indices.

    This class orchestrates:
      1) Target spectrum & conditioning at site/period (via `TargetSpectrum`).
      2) Per-record indices (epsilon(T), SaRatio) and regressions (via `SacIndexRelationship`).
      3) SSF from regression lines at target indices.
      4) Fragility curve median/dispersion and optional plotting.

    Attributes:
        # Site/structure inputs
        T (float): Fundamental period of the structure (s).
        longitude (float): Site longitude in degrees (W negative).
        latitude (float): Site latitude in degrees.
        Vs30 (float): Time-averaged shear-wave velocity in top 30 m (m/s).
        return_period (int | float): Return period for UHS (years).
        region (str): Region keyword for GMM; e.g., "california".
        mechanism (str): Faulting mechanism, one of {"U","SS","NS","RS"}.
        Z10 (float): Depth to Vs=1.0 km/s (km). Use -1 if unknown.

        # SaRatio window
        a (float): Lower factor; lower bound is a*T.
        b (float): Upper factor; upper bound is b*T.

        # Deaggregation vintage
        version (str): "2008" or "2023" for USGS APIs.

        # Data tables
        Sac_Form (pd.DataFrame): Two-column table with [EQ_ID (int), Sac (float)].

        # Objects produced during the pipeline
        target (TargetSpectrum): Target spectrum worker after `run_SSF()`.
        Sac_index_relationship (SacIndexRelationship): Index/Regression worker after `run_SSF()`.

        # Scalars produced during the pipeline
        SSF_epsilon (float): SSF from epsilon-based regression (ratio of shifted/original medians at the end).
        SSF_SaRatio (float): SSF from SaRatio-based regression (ratio of shifted/original medians at the end).

        # Fragility summaries (set by MedianDispersion* methods)
        fragility_epsilon (dict): For epsilon shift:
            {"Type":"Epsilon", "Median Sa Collapse":float, "Shifted Sa Collapse":float,
             "Dispersion Sa Collapse":float}.
        fragility_SaRatio (dict): For SaRatio shift:
            {"Type":"Sa Ratio", "Median Sa Collapse":float, "Shifted Sa Collapse":float,
             "Dispersion Sa Collapse":float}.

        # Optional plot cache (set in PlotFragilityCurve with Option="both")
        fragility_data (dict): {"Original":{"x":np.ndarray,"y":np.ndarray},
                                "Epsilon":{"x":np.ndarray,"y":np.ndarray},
                                "Sa_Ratio":{"x":np.ndarray,"y":np.ndarray}}.
    """
    def __init__(self,T,longitude,latitude,Vs30,return_period,a,b,version,Sac_Form,region="california",mechanism="SS",Z10=-1):
        """
        Initialize SSF workflow with site/period, data tables, and options.

        Args:
            T (float): Structure period (s).
            longitude (float): Longitude (deg).
            latitude (float): Latitude (deg).
            Vs30 (float): Site Vs30 (m/s).
            return_period (int | float): UHS return period (years).
            a (float): Lower factor for SaRatio window.
            b (float): Upper factor for SaRatio window.
            version (str): USGS deaggregation version, "2008" or "2023".
            Sac_Form (pd.DataFrame): Two-column DataFrame [EQ_ID (int), Sac (float)].
            region (str, optional): GMM region keyword. Default "california".
            mechanism (str, optional): Faulting mechanism {"U","SS","NS","RS"}. Default "SS".
            Z10 (float, optional): Depth to Vs=1.0 km/s (km). Use -1 if unknown. Default -1.

        Returns:
            None
        """
        self.T=T
        self.longitude=longitude
        self.latitude=latitude
        self.Vs30=Vs30
        self.return_period=return_period
        self.region=region
        self.mechanism=mechanism
        self.Z10=Z10
        self.a=a
        self.b=b
        self.version=version
        self.region=region
        self.mechanism=mechanism
        self.Z10=Z10
        self.Sac_Form=Sac_Form
    def SSFForEpsilon(self,Epsilon_Target,Result_File):
        """
        Compute SSF from epsilon-based regression result.

        Definition (as implemented):
            SSF = exp( beta[0] * (Epsilon_Target - mean(Epsilon_set)) ),
        where beta[0] is slope from regression ln(Sac) = beta[0]*epsilon + beta[1].

        Args:
            Epsilon_Target (float): Target epsilon at period T (from `TargetSpectrum`).
            Result_File (dict): Output of `SacIndexRelationship.regression_epsilon_Sac()`.
                Required keys:
                    - "Original Data": {"Epsilon": list[float], ...}
                    - "beta": list[float] with slope at index 0

        Returns:
            None

        Side Effects:
            Sets `self.SSF_epsilon` (float).
        """

        # calculate the mean value of the Epsilon from the dictionary
        Epsilon_Set=np.mean(Result_File["Original Data"]["Epsilon"])

        # calculate the SSF
        SSF=np.exp(Result_File["beta"][0]*(Epsilon_Target-Epsilon_Set))
        self.SSF_epsilon=SSF
        return
    def SSFForSaRatio(self,Sa_Ratio_Target,Result_File):
        """
        Compute SSF from SaRatio-based regression result.

        Definition (as implemented):
            SSF = beta[0] * (Sa_Ratio_Target - median(SaRatio_set)),
        where beta[0] is slope from regression Sac = beta[0]*SaRatio + beta[1].

        Args:
            Sa_Ratio_Target (float): Target SaRatio at period T (from `TargetSpectrum`).
            Result_File (dict): Output of `SacIndexRelationship.regression_SaRatio_Sac()`.
                Required keys:
                    - "Original Data": {"Sa Ratio": list[float], ...}
                    - "beta": list[float] with slope at index 0

        Returns:
            None

        Side Effects:
            Sets `self.SSF_SaRatio` (float).
        """
        # calculate the mean value of the Epsilon from the dictionary
        Sa_Ratio_Set=np.median(Result_File["Original Data"]["Sa Ratio"])
        SSF=Result_File["beta"][0]*(Sa_Ratio_Target-Sa_Ratio_Set)
        # calculate the SSF
        self.SSF_SaRatio=SSF
        return
        # Function to calculate theta and beta for the fragility curve
    def mle_fragility(self,IM, num_obs, num_fail):
        """
        Maximum-likelihood estimation of (theta, beta) for lognormal fragility with probit link.

        Model:
            P(collapse | IM) = Phi( a + b * ln(IM) ), where Phi is the standard normal CDF.
            Then:
                theta = exp(-a / b),   beta = 1 / b.

        Args:
            IM (array-like): Intensity measure values (>0); shape (n,) or broadcastable to (n,).
            num_obs (int | array-like): Total trials per IM bin; scalar or shape (n,).
            num_fail (array-like): Fail counts per IM bin; shape (n,).

        Returns:
            tuple[float, float]:
                - theta (float): Median (lognormal) of the fragility curve.
                - beta (float): Lognormal dispersion (std. dev. of ln(IM) at 50% collapse).

        Raises:
            ValueError: If array lengths mismatch or IM contains non-positive values.

        Notes:
            Uses statsmodels GLM with Binomial family and Probit link:
                Y = [fail, success], X = [1, ln(IM)].
        """
        IM = np.asarray(IM).reshape(-1)
        num_fail = np.asarray(num_fail).reshape(-1)

        if np.isscalar(num_obs):
            num_obs = np.ones_like(num_fail) * num_obs
        else:
            num_obs = np.asarray(num_obs).reshape(-1)

        num_success = num_obs - num_fail

        Y = np.column_stack((num_fail, num_success))

        X = sm.add_constant(np.log(IM))

        model = sm.GLM(Y, X, family=sm.families.Binomial(link=sm.families.links.Probit()))
        results = model.fit()

        b0, b1 = results.params
        theta = np.exp(-b0 / b1)
        beta = 1 / b1

        return theta, beta
    def MedianDispersionForEpsilonCurve(self):
        """
        Compute median & dispersion for epsilon-shifted fragility; store a summary dict.

        Procedure:
            1) Fit fragility to original Sac sample using `mle_fragility`.
            2) Shift the median by SSF_epsilon (multiplicative).
            3) Record {"Type":"Epsilon", "Median Sa Collapse", "Shifted Sa Collapse", "Dispersion Sa Collapse"}.

        Args:
            None

        Returns:
            None

        Side Effects:
            Sets `self.fragility_epsilon` (dict).

        Requires:
            - self.Sac_Form: two-column DataFrame [EQ_ID, Sac].
            - self.SSF_epsilon (float): set by `SSFForEpsilon`.
        """
        # Get the median and the dispersion for the Sac
        Median_Sac_Original,Dispersion_Sac_Original=self.mle_fragility(sorted(self.Sac_Form.iloc[:,1]),len(self.Sac_Form.iloc[:,1]),np.arange(1,self.Sac_Form.shape[0]+1))

        # Shift the median according to the SSF
        Median_Sac_Shift=self.SSF_epsilon*Median_Sac_Original

        # create the output dict
        Output= {
            "Type": "Epsilon",
            "Median Sa Collapse":Median_Sac_Original,
            "Shifted Sa Collapse":Median_Sac_Shift,
            "Dispersion Sa Collapse": Dispersion_Sac_Original
        }
        self.fragility_epsilon=Output
        return
    def MedianDispersionForSaRatioCurve(self):
        """
        Compute median & dispersion for SaRatio-shifted fragility; store a summary dict.

        Procedure:
            1) Fit fragility to original Sac sample using `mle_fragility`.
            2) Shift the median by SSF_SaRatio (additive, as implemented).
            3) Record {"Type":"Sa Ratio", "Median Sa Collapse", "Shifted Sa Collapse", "Dispersion Sa Collapse"}.

        Args:
            None

        Returns:
            None

        Side Effects:
            Sets `self.fragility_SaRatio` (dict).

        Requires:
            - self.Sac_Form: two-column DataFrame [EQ_ID, Sac].
            - self.SSF_SaRatio (float): set by `SSFForSaRatio`.
        """
        # Get the median and the dispersion for the Sac
        Median_Sac_Original,Dispersion_Sac_Original=self.mle_fragility(sorted(self.Sac_Form.iloc[:,1]),len(self.Sac_Form.iloc[:,1]),np.arange(1,self.Sac_Form.shape[0]+1))

        # Shift the median according to the SSF
        Median_Sac_Shift=self.SSF_SaRatio+Median_Sac_Original

        # create the output dict
        Output= {
            "Type": "Sa Ratio",
            "Median Sa Collapse":Median_Sac_Original,
            "Shifted Sa Collapse":Median_Sac_Shift,
            "Dispersion Sa Collapse": Dispersion_Sac_Original
        }
        self.fragility_SaRatio=Output
        return
    def PlotFragilityCurve(self,Paras,Option="both"):
        """
        Plot fragility curves for epsilon-shift, SaRatio-shift, or both.

        Args:
            Paras (list[dict]): List of parameter dicts as returned by
                `MedianDispersionForEpsilonCurve()` and/or `MedianDispersionForSaRatioCurve()`.
                Valid lists:
                    [epsilon_dict], [saratio_dict], or [epsilon_dict, saratio_dict] (order arbitrary).
            Option (str, optional): One of {"Epsilon","Sa Ratio","both"}; default "both".

        Returns:
            None
            (When Option="both", also sets `self.fragility_data` dict with x/y arrays.)

        Raises:
            ValueError: If `Paras` is not a list, or required dict for the chosen option is missing.

        Side Effects:
            Displays Matplotlib figure(s). With "both", stores plotted arrays in `self.fragility_data`.
        """
        # check the Paras is a list
        if type(Paras) != list:
            raise ValueError("The first input should be a list")
            return 0

        # the "Epsilon" option
        if Option == "Epsilon":
            # check the paras are correct
            for i in range(len(Paras)):
                if Paras[i]["Type"] == "Epsilon":
                    Paras_dict=Paras[i]
                    break
                if i == len(Paras)-1:
                    raise ValueError("No available parameters for Epsilon shift")
                    return 0
            # Calculate x and y for the plot
            x=np.arange(0.01,8,0.01)
            y_Original=norm.cdf((np.log(x)-np.log(Paras_dict["Median Sa Collapse"]))/Paras_dict["Dispersion Sa Collapse"])
            y_Epsilon=norm.cdf((np.log(x)-np.log(Paras_dict["Shifted Sa Collapse"]))/Paras_dict["Dispersion Sa Collapse"])
            # Plot the fragility curve
            fig,ax=plt.subplots()
            ax.plot(x,y_Original,label=r"IDA, $\theta$={},$\beta$={}".format(round(Paras_dict["Median Sa Collapse"],2),round(Paras_dict["Dispersion Sa Collapse"],2)))
            ax.plot(x,y_Epsilon,label=r"$\epsilon$, $\theta$={},$\beta$={}".format(round(Paras_dict["Shifted Sa Collapse"],2),round(Paras_dict["Dispersion Sa Collapse"],2)))
            ax.legend()
            ax.grid(True)
            ax.set_xlabel(r"$Sa$")
            ax.set_ylabel(r"P(c)")
            ax.set_title("Fragility Curves")
            plt.show()
        # the "Sa Ratio" option
        elif Option == "Sa Ratio":
            # check the paras are correct
            for i in range(len(Paras)):
                if Paras[i]["Type"] == "Sa Ratio":
                    Paras_dict=Paras[i]
                    break
                if i == len(Paras)-1:
                    raise ValueError("No available parameters for Sa Ratio shift")
                    return 0
            # Calculate x and y for the plot
            x=np.arange(0.01,8,0.01)
            y_Original=norm.cdf((np.log(x)-np.log(Paras_dict["Median Sa Collapse"]))/Paras_dict["Dispersion Sa Collapse"])
            y_Epsilon=norm.cdf((np.log(x)-np.log(Paras_dict["Shifted Sa Collapse"]))/Paras_dict["Dispersion Sa Collapse"])
            # Plot the fragility curve
            fig,ax=plt.subplots()
            ax.plot(x,y_Original,label=r"IDA, $\theta$={},$\beta$={}".format(round(Paras_dict["Median Sa Collapse"],2),round(Paras_dict["Dispersion Sa Collapse"],2)))
            ax.plot(x,y_Epsilon,label=r"$Sa\ Ratio$, $\theta$={},$\beta$={}".format(round(Paras_dict["Shifted Sa Collapse"],2),round(Paras_dict["Dispersion Sa Collapse"],2)))
            ax.legend()
            ax.grid(True)
            ax.set_xlabel(r"$Sa$")
            ax.set_ylabel(r"P(c)")
            ax.set_title("Fragility Curves")
            plt.show()
        # the "both" option
        elif Option=="both":
            # first plot the Epsilon
            # check the paras are correct
            for i in range(len(Paras)):
                if Paras[i]["Type"] == "Epsilon":
                    Paras_dict=Paras[i]
                    break
                if i == len(Paras)-1:
                    raise ValueError("No available parameters for Epsilon shift")
                    return 0
            # Calculate x and y for the plot
            x=np.arange(0.01,8,0.01)
            y_Original=norm.cdf((np.log(x)-np.log(Paras_dict["Median Sa Collapse"]))/Paras_dict["Dispersion Sa Collapse"])
            y_Epsilon=norm.cdf((np.log(x)-np.log(Paras_dict["Shifted Sa Collapse"]))/Paras_dict["Dispersion Sa Collapse"])
            # Plot the fragility curve
            fig,ax=plt.subplots()
            ax.plot(x,y_Original,label=r"IDA, $\theta$={},$\beta$={}".format(round(Paras_dict["Median Sa Collapse"],2),round(Paras_dict["Dispersion Sa Collapse"],2)))
            ax.plot(x,y_Epsilon,label=r"$\epsilon$, $\theta$={},$\beta$={}".format(round(Paras_dict["Shifted Sa Collapse"],2),round(Paras_dict["Dispersion Sa Collapse"],2)))

            # then plot the Sa Ratio
            # check the paras are correct
            for i in range(len(Paras)):
                if Paras[i]["Type"] == "Sa Ratio":
                    Paras_dict=Paras[i]
                    break
                if i == len(Paras)-1:
                    raise ValueError("No available parameters for Sa Ratio shift")
                    return 0
            # Calculate x and y for the plot
            y_SaRatio=norm.cdf((np.log(x)-np.log(Paras_dict["Shifted Sa Collapse"]))/Paras_dict["Dispersion Sa Collapse"])
            # Plot the fragility curve
            ax.plot(x,y_SaRatio,label=r"$Sa\ Ratio$, $\theta$={},$\beta$={}".format(round(Paras_dict["Shifted Sa Collapse"],2),round(Paras_dict["Dispersion Sa Collapse"],2)))
            ax.legend()
            ax.grid(True)
            ax.set_xlabel(r"$Sa$")
            ax.set_ylabel(r"P(c)")
            ax.set_title("Fragility Curves")
            output={"Original":{"x":x,"y":y_Original},"Epsilon":{"x":x,"y":y_Epsilon},"Sa_Ratio":{"x":x,"y":y_SaRatio}}
            plt.show()
            self.fragility_data=output
            return
        else:
            raise ValueError("No such option, availbale options:'Epsilon','Sa Ratio','both'")
    def run_SSF(self):
        """
        End-to-end SSF workflow: target → indices/regressions → SSFs → fragility → plots.

        Steps:
            1) Build `TargetSpectrum` and compute:
               UHS → target epsilon → CMS → target SaRatio.
            2) Build `SacIndexRelationship` and compute:
               epsilon(T) per record, SaRatio per record, both regressions.
            3) Compute SSF from both regressions at target indices.
            4) Fit fragilities and compute shifted medians/dispersion.
            5) Convert to display SSFs as ratios of shifted/original medians.
            6) Plot fragility curves (both).

        Args:
            None

        Returns:
            None

        Side Effects:
            Populates: `self.target`, `self.Sac_index_relationship`,
            `self.SSF_epsilon`, `self.SSF_SaRatio`,
            `self.fragility_epsilon`, `self.fragility_SaRatio`,
            and displays plots via `PlotFragilityCurve`.
        """
        # create the target
        self.target=TargetSpectrum(self.T,self.longitude,self.latitude,self.Vs30,self.return_period,self.a,self.b,self.version,self.region,self.mechanism,self.Z10)
        self.target.run_target()
        # create the Sac-epsilon/SaRatio
        self.Sac_index_relationship=SacIndexRelationship(self.T,self.a,self.b,self.Sac_Form)
        self.Sac_index_relationship.run_Sac_index()
        # calculate the SSF
        print("="*50)
        print("Start of SSF calculation section")
        self.SSFForEpsilon(self.target.target_epsilon,self.Sac_index_relationship.epislon_regression_result)
        self.SSFForSaRatio(self.target.target_SaRatio,self.Sac_index_relationship.SaRatio_regression_result)
        # get the fragility curve info
        self.MedianDispersionForEpsilonCurve()
        self.MedianDispersionForSaRatioCurve()
        # calculate the SSF to present
        self.SSF_epsilon=self.fragility_epsilon["Shifted Sa Collapse"]/self.fragility_epsilon["Median Sa Collapse"]
        self.SSF_SaRatio=self.fragility_SaRatio["Shifted Sa Collapse"]/self.fragility_SaRatio["Median Sa Collapse"]
        print("-"*50)
        print(" "*14,"SSF and Fragility Curve"," "*14)
        print("SSF (epsilon): ",self.SSF_epsilon)
        print("SSF (Sa Ratio): ",self.SSF_SaRatio)
        print("Median Sac (wihtout shift): ",self.fragility_epsilon["Median Sa Collapse"], " g")
        print("Median Sac (epsilon shift): ",self.fragility_epsilon["Shifted Sa Collapse"], " g")
        print("Median Sac (Sa Ratio shift): ",self.fragility_SaRatio["Shifted Sa Collapse"], " g")
        print("Dispersion: ",self.fragility_SaRatio["Dispersion Sa Collapse"], " g")
        print("-"*50)
        print("Plots")
        self.PlotFragilityCurve([self.fragility_epsilon,self.fragility_SaRatio])
        print("-"*50)
        print("End of SSF calculation section")
        print("="*50)

# Class for calculating Hazard Curve
class HazardCurve:
    """
    Retrieve and interpolate USGS hazard curves at a site and target period T.

    Attributes:
        T (float): Target oscillator period for which to obtain the hazard curve (s).
        longitude (float): Site longitude (deg, west negative). USGS APIs typically expect [-125, -65] (CONUS).
        latitude (float): Site latitude (deg). USGS APIs typically expect [24.4, 50] (CONUS).
        Vs30 (float): Time-averaged shear-wave velocity at 30 m (m/s).
        version (str): USGS hazard vintage, one of {"2008", "2023"}.
        Tlist (np.ndarray): Dense Sa grid used for interpolating / plotting (0.01–8.0 s step 0.01).
        harzrd_curve_api (list[dict]): Raw hazard-curves per IMT from API; each dict has:
            {"Period": float, "x": list[float], "y": list[float]} with x=Sa [g], y=annual rate.
            Populated by `HazardCurve08()` or `HazardCurve23()`.
        hazard_curve (dict): Interpolated hazard curve at period T:
            {"x": np.ndarray (Sa grid), "y": list[float] (annual exceedance)}.
            Set by `HazardCurveT()`.
    """
    def __init__(self,T,longitude,latitude,Vs30,version):
        """
        Initialize site, spectral period, Vs30 and USGS vintage.

        Args:
            T (float): Target period in seconds.
            longitude (float): Longitude (deg).
            latitude (float): Latitude (deg).
            Vs30 (float): Site Vs30 (m/s).
            version (str): "2008" (E2008/WUS endpoints) or "2023" (CONUS 2023 dynamic API).

        Raises:
            ValueError: If `version` is not one of {"2008","2023"}.

        Side Effects:
            Prints a short section header to stdout.
        """
        print("="*50)
        print("Start of hazard curve section")
        print("-"*50)
        self.T=T
        self.longitude=longitude
        self.latitude=latitude
        self.Vs30=Vs30
        self.version=version
        self.Tlist=np.arange(0.01,8.01,0.01)
        if version=="2008":
            print("You are using 2008 USGS Hazard")
        elif version=="2023":
            print("You are using 2023 USGS Hazard")
        else:
            raise ValueError("Only '2008' and '2023' versions are supported")
    def HazardCurve08(self):
        """
        Fetch hazard curves (Sa vs annual exceedance) from the USGS 2008 WUS API.

        Queries fixed IMTs corresponding to periods:
            [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 3.0] s.
        Stores each curve as {"Period": P, "x": xs, "y": ys} into `self.harzrd_curve_api`.

        Args:
            None

        Returns:
            None

        Side Effects:
            Sets `self.harzrd_curve_api` (list of dicts) and prints status.

        Raises:
            requests.RequestException / ValueError:
                If the HTTP request fails or response structure is unexpected.
        """
        print("-"*50)
        print("start running API to get USGS Deaggregation.")
        Period=[0.1,0.2,0.3,0.5,0.75,1.0,2.0,3.0]
        imts=["SA0P1","SA0P2","SA0P3","SA0P5","SA0P75","SA1P0","SA2P0","SA3P0"]
        Output=[]
        for i in range(len(imts)):
            url=f"https://earthquake.usgs.gov/nshmp-haz-ws/hazard/E2008/WUS/{self.longitude}/{self.latitude}/{imts[i]}/{self.Vs30}"
            response=requests.get(url)
            data=response.json()
            HC_x=data["response"][0]["metadata"]["xvalues"]
            HC_y=data["response"][0]['data'][0]['yvalues']
            Output_Period={'Period':Period[i],'x':HC_x,'y':HC_y}
            Output.append(Output_Period)
        self.harzrd_curve_api=Output
        print("Hazard curve data done.")
        return
    def HazardCurve23(self):
        """
        Fetch hazard curves (Sa vs annual exceedance) from the USGS 2023 CONUS dynamic API.

        Queries multiple IMTs in a single request, covering periods:
            [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3,
             0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0] s.
        Stores each curve as {"Period": P, "x": xs, "y": ys} into `self.harzrd_curve_api`.

        Args:
            None

        Returns:
            None

        Side Effects:
            Sets `self.harzrd_curve_api` (list of dicts) and prints status.

        Raises:
            requests.RequestException / ValueError:
                If the HTTP request fails or response structure is unexpected.
        """
        print("-"*50)
        print("start running API to get USGS Deaggregation.")
        url=f"https://earthquake.usgs.gov/ws/nshmp/conus-2023/dynamic/hazard/{self.longitude}/{self.latitude}/{self.Vs30}?truncate=true&maxdir=false&imt=SA0P01&imt=SA0P02&imt=SA0P03&imt=SA0P05&imt=SA0P075&imt=SA0P1&imt=SA0P15&imt=SA0P2&imt=SA0P25&imt=SA0P3&imt=SA0P4&imt=SA0P5&imt=SA0P75&imt=SA1P0&imt=SA1P5&imt=SA2P0&imt=SA3P0&imt=SA4P0&imt=SA5P0&imt=SA7P5&imt=SA10P0"
        response=requests.get(url)
        data=response.json()
        Period=[0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1.0,1.5,2.0,3.0,4.0,5.0,7.5,10.0]
        Output=[]
        for i in range(len(Period)):
            HC_x=data["response"]["hazardCurves"][i]["data"][0]["values"]['xs']
            HC_y=data["response"]["hazardCurves"][i]["data"][0]["values"]['ys']
            Output_Period={'Period':Period[i],'x':HC_x,'y':HC_y}
            Output.append(Output_Period)
        self.harzrd_curve_api=Output
        print("Hazard curve data done.")
        return
    def HazardCurveT(self):
        """
        Interpolate a hazard curve at target period `self.T` from two neighboring API curves.

        Method:
            1) Locate periods P_low < T < P_up from `self.harzrd_curve_api`.
            2) Remove zero y-values (to avoid log issues).
            3) Fit degree-6 polynomials in log-log space separately to (P_low) and (P_up) curves.
            4) Evaluate both fits on `self.Tlist` Sa-grid; log-linear interpolate across period to T.
            5) Store and (log-log) plot the resulting curve.

        Args:
            None

        Returns:
            dict: Hazard curve at T, {"x": self.Tlist (np.ndarray of Sa), "y": list[float] (annual rate)}.

        Side Effects:
            Sets `self.hazard_curve` and displays a Matplotlib log-log plot.

        Requires:
            `self.harzrd_curve_api` must be populated by `HazardCurve08()` or `HazardCurve23()`.

        Notes:
            - Comment mentions 4th-order, but the implementation uses degree=6 polynomial fits.
            - x = Sa [g], y = annual rate of exceedance; both axes are plotted in log scale.
        """
        for i in range(len(self.harzrd_curve_api)):
            if self.harzrd_curve_api[i]["Period"]>self.T:
                Hazard_Curve_Up=self.harzrd_curve_api[i]
                Hazard_Curve_Low=self.harzrd_curve_api[i-1]
                nonzero_indices_Up = [j for j, x in enumerate(Hazard_Curve_Up["y"]) if x != 0]
                Hazard_Curve_Up["x"]=[x for j, x in enumerate(Hazard_Curve_Up["x"]) if j in nonzero_indices_Up]
                Hazard_Curve_Up["y"]=[y for j, y in enumerate(Hazard_Curve_Up["y"]) if j in nonzero_indices_Up]
                nonzero_indices_Low = [j for j, x in enumerate(Hazard_Curve_Low["y"]) if x != 0]
                Hazard_Curve_Low["x"]=[x for j, x in enumerate(Hazard_Curve_Low["x"]) if j in nonzero_indices_Low]
                Hazard_Curve_Low["y"]=[y for j, y in enumerate(Hazard_Curve_Low["y"]) if j in nonzero_indices_Low]
                # do the 4th order polynomial fit
                Hazard_Curve_Up_coeff=np.polyfit(np.log(Hazard_Curve_Up['x']),np.log(Hazard_Curve_Up['y']),deg=6)
                Hazard_Curve_Up_func=np.poly1d(Hazard_Curve_Up_coeff)
                Hazard_Curve_Up_y=np.exp(Hazard_Curve_Up_func(np.log(self.Tlist)))
                Hazard_Curve_Low_coeff=np.polyfit(np.log(Hazard_Curve_Low['x']),np.log(Hazard_Curve_Low['y']),deg=6)
                Hazard_Curve_Low_func=np.poly1d(Hazard_Curve_Low_coeff)
                Hazard_Curve_Low_y=np.exp(Hazard_Curve_Low_func(np.log(self.Tlist)))
                # interpolate to get the result
                Hazard_Curve_T_y=[]
                for j in range(len(self.Tlist)):
                    Hazard_Curve_T_y.append(np.exp(np.interp(self.T,[Hazard_Curve_Low["Period"],Hazard_Curve_Up["Period"]],[np.log(Hazard_Curve_Low_y[j]),np.log(Hazard_Curve_Up_y[j])])))
                Hazard_Curve_T={'x':self.Tlist,'y':Hazard_Curve_T_y}
                #plt.plot(Hazard_Curve_T["x"],Hazard_Curve_Up_y,label=r"{} fit".format(Hazard_Curve_Up["Period"]))
                #plt.plot(Hazard_Curve_T["x"],Hazard_Curve_Low_y,label=r"{} fit".format(Hazard_Curve_Low["Period"]))
                #plt.plot(Hazard_Curve_Up["x"],Hazard_Curve_Up["y"],label=r"{}".format(Hazard_Curve_Up["Period"]))
                #plt.plot(Hazard_Curve_Low["x"],Hazard_Curve_Low["y"],label=r"{}".format(Hazard_Curve_Low["Period"])) 
                plt.plot(Hazard_Curve_T["x"],Hazard_Curve_T["y"],label="result")
                plt.grid(True)
                plt.xlabel("Sa [g]")
                plt.ylabel("Annual Rate of Exceedance")
                plt.xscale("log")
                plt.yscale("log")
                plt.title(f"Hazard Curve for T={round(self.T,2)} s")
                #plt.legend()
                plt.show()
                break
        self.hazard_curve=Hazard_Curve_T
        return Hazard_Curve_T
    def run_hazard_curve(self):
        """
        End-to-end hazard curve workflow for target period `T`.

        Steps:
            1) Fetch API hazard curves:
               - `HazardCurve08()` if version == "2008"
               - `HazardCurve23()` if version == "2023"
            2) Interpolate to period T and plot via `HazardCurveT()`.

        Args:
            None

        Returns:
            None

        Side Effects:
            Populates `self.harzrd_curve_api` and `self.hazard_curve`; displays a plot and prints status.

        Raises:
            ValueError: If `version` is unsupported.
        """
        if self.version=="2008":
            self.HazardCurve08()
        elif self.version=="2023":
            self.HazardCurve23()
        else:
            raise ValueError("Only '2008' and '2023' versions are supported")
        print("Plots:")
        self.HazardCurveT()
        print("-"*50)
        print("End of hazard curve section")
        print("="*50)

# Class to calculate the MAF
class MAF:
    """
    Compute the Mean Annual Frequency (MAF) of failure from hazard and fragility curves.

    Attributes:
        hazard_curve (dict): Hazard curve data, typically of the form:
            {
                "x": np.ndarray or list[float],  # Sa values [g]
                "y": np.ndarray or list[float],  # Annual rate of exceedance
            }
            The curve should be defined in descending order of Sa (x increasing, y decreasing).

        fragility_curve (dict): Fragility function data, typically of the form:
            {
                "x": np.ndarray or list[float],  # Sa values [g], same range as hazard_curve['x']
                "y": np.ndarray or list[float],  # Probability of collapse at each Sa
            }

        lambda_failure (float): Mean annual frequency of failure (computed in `MAF()`).

    Notes:
        The MAF is computed using numerical integration:
            λ_failure = ∫ P(collapse | Sa) * dλ_exceedance(Sa)
        which is approximated by a trapezoidal sum over discrete points.
    """
    def __init__(self,hazard_curve,fragility_curve):
        """
        Initialize with hazard and fragility curve data.

        Args:
            hazard_curve (dict): Dictionary containing hazard curve values,
                with keys 'x' (Sa [g]) and 'y' (annual exceedance rate).
            fragility_curve (dict): Dictionary containing fragility curve values,
                with keys 'x' (Sa [g]) and 'y' (probability of collapse).

        Returns:
            None

        Raises:
            ValueError: If either input does not contain the required keys.
        """
        self.hazard_curve=hazard_curve
        self.fragility_curve=fragility_curve
    def MAF(self):
        """
        Compute the mean annual frequency of failure (λ_failure).

        Integration method:
            λ_failure = Σ [ (P_i + P_{i+1}) / 2 ] * [ (λ_i - λ_{i+1}) ]
        where:
            P_i = fragility probability at Sa_i,
            λ_i = hazard exceedance rate at Sa_i.

        Args:
            None

        Returns:
            None

        Side Effects:
            Sets `self.lambda_failure` (float): mean annual frequency of failure [1/year].
        """
        lambda_failure=0
        for i in range(len(self.fragility_curve['x'])-1):
            lambda_failure+=(self.fragility_curve['y'][i]+self.fragility_curve['y'][i+1])/2*(self.hazard_curve['y'][i]-self.hazard_curve['y'][i+1])
        self.lambda_failure=lambda_failure
        return
    def run_MAF(self):
        """
        Execute the MAF computation workflow.

        Runs:
            1) `MAF()` — numerical integration of hazard * fragility curves.
            2) Stores the result in `self.lambda_failure`.

        Args:
            None

        Returns:
            None

        Side Effects:
            Populates `self.lambda_failure` with the computed mean annual frequency of failure.
        """
        self.MAF()

# Class to go through the whole procedure
class FEMAP695SSF:
    """
    End-to-end pipeline for FEMA P-695 style Spectral Shape Factor (SSF) and MAF.

    This orchestrates:
      1) SSF workflow (target spectra → indices/regressions → SSFs → fragility),
      2) Hazard curve retrieval & interpolation,
      3) Mean Annual Frequency (MAF) of collapse with/without shape shifts.

    Attributes:
        # Site/structure inputs
        T (float): Fundamental period of the structure (s).
        longitude (float): Site longitude in degrees (west negative).
        latitude (float): Site latitude in degrees.
        Vs30 (float): Time-averaged shear-wave velocity in the top 30 m (m/s).
        return_period (int | float): Return period for UHS (years).

        # GMM/region options
        region (str): Region keyword for GMM, e.g., "california".
        mechanism (str): Faulting mechanism, one of {"U","SS","NS","RS"}.
        Z10 (float): Depth to Vs=1.0 km/s (km). Use -1 if unknown.

        # SaRatio window
        a (float): Lower factor for the period window a*T.
        b (float): Upper factor for the period window b*T.

        # USGS deaggregation vintage
        version (str): "2008" or "2023".

        # Data tables
        Sac_Form (pd.DataFrame): Two-column table with [EQ_ID (int), Sac (float)].

        # Sub-objects populated by `run()`
        ssf (SSF): SSF/fragility calculation object (after `run()`).
        hazard_curve (HazardCurve): Hazard-curve object (after `run()`).
        maf (MAF): MAF object used for the last printed result (after `run()`).

        # Convenience accessors (properties)
        target_epsilon (float): Target ε(T) from `ssf.target`.
        target_Sa_Ratio (float): Target SaRatio from `ssf.target`.
        CMS (dict): Conditional Mean Spectrum results from `ssf.target`.
        SSF_epsilon (float): Final SSF (epsilon-based), ratio of shifted/original medians.
        SSF_Sa_Ratio (float): Final SSF (SaRatio-based), ratio of shifted/original medians.
    """
    def __init__(self,T,longitude,latitude,Vs30,return_period,a,b,version,Sac_Form,region="california",mechanism="SS",Z10=-1):
        """
        Initialize pipeline inputs and data.

        Args:
            T (float): Structure period (s).
            longitude (float): Longitude (deg).
            latitude (float): Latitude (deg).
            Vs30 (float): Site Vs30 (m/s).
            return_period (int | float): UHS return period (years).
            a (float): Lower factor for SaRatio window (a*T).
            b (float): Upper factor for SaRatio window (b*T).
            version (str): USGS deaggregation version, "2008" or "2023".
            Sac_Form (pd.DataFrame): Two-column DataFrame [EQ_ID (int), Sac (float)].
            region (str, optional): GMM region keyword. Default "california".
            mechanism (str, optional): Faulting mechanism {"U","SS","NS","RS"}. Default "SS".
            Z10 (float, optional): Depth to Vs=1.0 km/s (km). -1 if unknown. Default -1.

        Returns:
            None
        """
        self.T=T
        self.longitude=longitude
        self.latitude=latitude
        self.Vs30=Vs30
        self.return_period=return_period
        self.region=region
        self.mechanism=mechanism
        self.Z10=Z10
        self.a=a
        self.b=b
        self.version=version
        self.region=region
        self.mechanism=mechanism
        self.Z10=Z10
        self.Sac_Form=Sac_Form

    #run the whole analysis
    def run(self):
        """
        Execute the full FEMA P-695 style workflow and print summary results.

        Steps:
            1) Build and run SSF pipeline:
               - target spectra (UHS, ε(T), CMS, SaRatio),
               - record-wise indices & regressions,
               - SSF (epsilon & SaRatio),
               - fragility medians/dispersion and curves (stored for plotting/MAF).
            2) Build and run hazard-curve pipeline:
               - fetch API curves (per vintage),
               - interpolate to target period T.
            3) Compute and print MAF for:
               - Original fragility (no shift),
               - Epsilon-shifted fragility,
               - SaRatio-shifted fragility.

        Args:
            None

        Returns:
            None

        Side Effects:
            - Populates `self.ssf`, `self.hazard_curve`, and `self.maf`.
            - Prints SSF and MAF results.
            - Generates plots within sub-components (hazard curve, fragility curves).
        """
        print("FEMA P-695 Spectural Shape Factor Calculation")
        self.ssf=SSF(self.T,self.longitude,self.latitude,self.Vs30,self.return_period,self.a,self.b,self.version,self.Sac_Form,self.region,self.mechanism,self.Z10)
        self.ssf.run_SSF()
        self.hazard_curve=HazardCurve(self.T,self.longitude,self.latitude,self.Vs30,self.version)
        self.hazard_curve.run_hazard_curve()
        print("="*50)
        print("Start of MAF calculation section")
        print("-"*50)
        self.maf=MAF(self.hazard_curve.hazard_curve,self.ssf.fragility_data["Original"])
        self.maf.run_MAF()
        print("MAF (without shift): ",self.maf.lambda_failure)
        self.maf=MAF(self.hazard_curve.hazard_curve,self.ssf.fragility_data["Epsilon"])
        self.maf.run_MAF()
        print("MAF (epsilon shift): ",self.maf.lambda_failure)
        self.maf=MAF(self.hazard_curve.hazard_curve,self.ssf.fragility_data["Sa_Ratio"])
        self.maf.run_MAF()
        print("MAF (Sa Ratio shift): ",self.maf.lambda_failure)
        print("-"*50)
        print("End of MAF calculation section")
        print("="*50)
    @property
    
    def target_epsilon(self): 
        """float: Target epsilon at period T computed within the SSF target workflow."""
        return self.ssf.target.target_epsilon 
    @property
    def target_Sa_Ratio(self): 
        """float: Target Sa ratio at period T computed within the SSF target workflow."""
        return self.ssf.target.target_SaRatio
    @property
    def CMS(self): 
        """dict: Conditional Mean Spectrum results from the SSF target workflow.

        Returns:
            dict with keys:
                - "median": np.ndarray of median CMS Sa (g) over the internal T grid,
                - "std ln(Sa)": np.ndarray of conditional ln standard deviations.
        """
        return self.ssf.target.CMS
    @property
    def SSF_epsilon(self): 
        """float: Final SSF based on epsilon regression (ratio of shifted/original medians)."""
        return self.ssf.SSF_epsilon
    @property
    def SSF_Sa_Ratio(self): 
        """float: Final SSF based on SaRatio regression (ratio of shifted/original medians)."""
        return self.ssf.SSF_SaRatio

        
        




