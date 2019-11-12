L8 Angles Readme 
--------------------------------------------------------------------------------
Making code:
    1. In the root l8_angles directory run the command:
        make

    2. The executable should now be present in the root directory.

    3. Test the executable by running the following:
        ./l8_angles --help

    4. The output should be as follows:

      Usage: l8_angles
         <MetadataFilename>: (Required) Angle coefficient filename

         <AngleType>: (Required) The type of angles to generate
                                 VALID: (BOTH, SATELLITE, SOLAR)

         <SubsampleFactor>: (Required) Sub-sample factor used when calculating
                                       the angles (integer)

         -f <FillPixelValue>: (Optional) Fill pixel value to use (short int)
                                         units used is degrees scaled by 100
                                         Default: 0
                                         Range: (-32768:32767)

         -b <BandList>: (Optional) Band list used to calculate angles for, this
                                defaults to all bands 1 - 11. Must be comma
                                separated with no spaces in between.
                                Example: 1,2,3,4,5,6,7,8,9
