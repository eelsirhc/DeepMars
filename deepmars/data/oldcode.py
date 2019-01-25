#@data.command()
#def craters_tsv_to_hdf(filename=None,output=None):
#    """Reads the crater database TSV and outputs to HDF.
#
#    Parameters
#    -----------
#    filename : str, optional
#        path of the TSV file, defaults to the value found in the environment variable DM_CraterTable
#    output : str, optional
#        path to the HDF file, defaults to the value found in the environment variable DM_CraterHDF
#    
#    Returns
#    --------
#    None
#    """
#
#    if filename is None:
#        filename = os.getenv("DM_CraterTable")
#    
#    craters = pd.read_table(filename,sep='\t',engine='python')
#    if output is None:
#        output = os.getenv("DM_CraterHDF")
#    print(output)
#    keep_columns = ["CRATER_ID",
#                    "LATITUDE_CIRCLE_IMAGE",
#                    "LONGITUDE_CIRCLE_IMAGE",
#                    "DIAM_CIRCLE_IMAGE",
#                    "DIAM_CIRCLE_SD_IMAGE",
#                    "DEPTH_RIM_TOPOG",
#                    "DEPTH_RIM_SD_TOPOG",
#                    "DEPTH_SURFACE_TOPOG",
#                    "DEPTH_SURFACE_SD_TOPOG",
#                    "DEPTH_FLOOR_TOPOG",
#                    "DEPTH_FLOOR_SD_TOPOG",
#                    "CRATER_NAME"]
#
#    print(craters[keep_columns].head())
#    craters[keep_columns].fillna('').to_hdf(output,key='/craters',nan_rep="nan")
    
