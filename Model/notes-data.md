# DataSets

[x] ERA5-Land

[x] Terrain Parameters (Slope and Aspect)

These are derived from Digital Elevation Models (DEMs) and are vital for understanding how topography influences fire spread.
> Link: https://bhoonidhi.nrsc.gov.in/

[ ] Thematic Data (Fuel Availability using LULC)

Land Use Land Cover (LULC) maps are fundamental for identifying different vegetation types, which serve as fuel for fires.
Bhuvan (NRSC/ISRO)
> Link: https://bhuvan.nrsc.gov.in/, https://bhuvan-app1.nrsc.gov.in/thematic/

[ ] Historical Fire Data

This data will serve as your target variable for training the fire prediction model and for validating the fire spread simulations.

    VIIRS-SNP (Visible Infrared Imaging Radiometer Suite - Suomi National Polar-orbiting Partnership)

        Data Available: The VIIRS I-Band 375m Active Fire Product provides data from the VIIRS sensor aboard the Suomi NPP and NOAA-20/21 satellites. This product offers improved spatial resolution over fires compared to MODIS, providing better mapping of large fire perimeters. Data is available from January 2012 to present for Suomi NPP.

        Access: Active fire/hotspot information can be downloaded from FIRMS (Fire Information for Resource Management System) in shapefile, CSV, or JSON formats. Authentication through Earthdata Login or email is required.

        Link: https://firms.modaps.eosdis.nasa.gov/download/

    Forest Survey of India (FSI)

        Data Available: FSI provides fire alerts and historical fire points detected by MODIS and SNPP sensors. You can search for fire points by state (including Madhya Pradesh) and date range.

        Access: Available directly on their website.

        Link: https://www.fsiforestfire.gov.in/

[ ] Human Settlement & Stressor Layers

These layers help identify potential ignition sources and areas of human impact, which are indirect factors influencing fire occurrence.

    GHSL (Global Human Settlement Layer)

        Data Available: GHSL provides gridded data on human population (GHS-POP) and built-up area (GHS-BUILT) for various time periods (e.g., 1975, 1990, 2000, 2014/2015). GHS-BUILT describes the percent built-up area for each grid cell, with resolutions available at 10m, 100m, 1km, 3 arcsec, and 30 arcsec. GHS-POP provides population counts at resolutions like 100m, 3 arc-seconds, and 30 arc-seconds.

        Access: Data can be directly downloaded from the GHSL website, often by tiles or as global datasets. Some GHSL products are also available through platforms like data.humdata.org.

        Link: https://ghsl.jrc.ec.europa.eu/download.php

Once you have these datasets, the next crucial step will be to preprocess them, ensuring they are all aligned to a common coordinate system and resampled to your target 30m resolution, forming a comprehensive feature stack for your machine learning models.
