;   Example script to produce plots for a WRF real-data run,
;   with the ARW coordinate dynamics option.
;   Interpolating to specified pressure levels

load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRFUserARW.ncl"

begin
;
; The WRF ARW input file.  
; This needs to have a ".nc" appended, so just do it.
  a = addfile("../../../../sf_sfclay_phys=1_isfflx=0/wrfout_d02_2023-08-20_06:00:00" + ".nc","r")


; We generate plots, but what kind do we prefer?
;  type = "x11"
  type = "png"
; type = "pdf"
; type = "ps"
; type = "ncgm"
;  wks = gsn_open_wks(type,"plt_PressureLevel1")
type@wkWidth = 3000
type@wkHeight = 3000


; Set some Basic Plot options
  res = True
  res@MainTitle                   = "REAL-TIME WRF"
  res@Footer = False

  pltres = True
  mpres = True
  mpres@mpOutlineBoundarySets  = "GeophysicalAndUSStates" ;"AllBoundaries"
  mpres@mpGridAndLimbOn             = False
  mpres@mpGeophysicalLineColor      = "Black"
  mpres@mpNationalLineColor         = "Black"
  mpres@mpUSStateLineColor          = "Black"
  mpres@mpGridLineColor             = "Black"
  mpres@mpLimbLineColor             = "Black"
  mpres@mpPerimLineColor            = "Black"
  mpres@mpGeophysicalLineThicknessF = 7.0
  mpres@mpGridLineThicknessF        = 7.0
  mpres@mpLimbLineThicknessF        = 7.0
  mpres@mpNationalLineThicknessF    = 7.0
  mpres@mpUSStateLineThicknessF     = 7.0

        color = read_colormap_file("davos")   ; read the color map

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; What times and how many time steps are in the data set?
  times = wrf_user_getvar(a,"times",-1)  ; get all times in the file
  ntimes = dimsizes(times)         ; number of times in the file

; The specific pressure levels that we want the data interpolated to.
  pressure_levels = (/ 700., 600. /) ; 500., 300./)   ; pressure levels to plot
  nlevels         = dimsizes(pressure_levels)     ; number of pressure levels

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  do it = 0,ntimes-1 ;, ;2             ; TIME LOOP

    print("Working on time: " + times(it) )
    res@TimeLabel = times(it)   ; Set Valid time to use on plots

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; First get the variables we will need        
	qv = wrf_user_getvar(a,"QVAPOR",it)


    tc = wrf_user_getvar(a,"tc",it)        ; T in C
    u  = wrf_user_getvar(a,"ua",it)        ; u averaged to mass points
    v  = wrf_user_getvar(a,"va",it)        ; v averaged to mass points
    p  = wrf_user_getvar(a, "pressure",it) ; pressure is our vertical coordinate
    z  = wrf_user_getvar(a, "z",it)        ; grid point height
    rh = wrf_user_getvar(a,"rh",it)        ; relative humidity

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

    do level = 0,nlevels-1                 ; LOOP OVER LEVELS

      pressure = pressure_levels(level)

   wks = gsn_open_wks(type,"Qv_Wind_" + pressure + "hPa_time_" + times(it)) 

	qv_plane = wrf_user_intrp3d(qv,p,"h",pressure,0.,False)
      tc_plane = wrf_user_intrp3d(tc,p,"h",pressure,0.,False)
      z_plane  = wrf_user_intrp3d( z,p,"h",pressure,0.,False)
      rh_plane = wrf_user_intrp3d(rh,p,"h",pressure,0.,False)
      u_plane  = wrf_user_intrp3d( u,p,"h",pressure,0.,False)
      v_plane  = wrf_user_intrp3d( v,p,"h",pressure,0.,False)

      spd     = (u_plane*u_plane + v_plane*v_plane)^(0.5) ; m/sec
      spd@description = "Wind Speed"
      spd@units = "m/s"
   ;   u_plane = u_plane*1.94386     ; kts
   ;   v_plane = v_plane*1.94386     ; kts
   ;   u_plane@units = "kts"
   ;   v_plane@units = "kts"

      ; Plotting options for Water Vapor
        opts = res
        opts@ContourParameters = (/ 0., .014, .002/)
	opts@cnFillPalette        = color(::-1,:) 
;        opts@cnFillPalette        = "WhiteBlueGreenYellowRed"
        opts@cnSpanFillPalette    = True
        opts@cnFillOn             = True
        opts@cnLinesOn            = False            ;-- draw contour lines
        contour_qv = wrf_contour(a,wks,qv_plane,opts)
;       contour_qv = wrf_contour(a,wks,QV_zoom,opts)
	delete(opts)



      ; Plotting options for Wind Speed                
        opts = res                          
        opts@cnLineColor = "MediumSeaGreen"
        opts@ContourParameters = (/ 10. /)
        opts@cnInfoLabelOrthogonalPosF = 0.07  ; offset second label information
        opts@gsnContourLineThicknessesScale = 3.0
        contour_spd = wrf_contour(a,wks,spd,opts)
        delete(opts)


      ; Plotting options for Wind Vectors                 
        opts = res          
        opts@FieldTitle = "Wind"   ; overwrite Field Title
        opts@NumVectors = 17       ; wind barb density
        opts@vcWindBarbLineThicknessF = 7
        opts@vcRefLengthF = 0.03
        vector = wrf_vector(a,wks,u_plane,v_plane,opts)
;        vector = wrf_vector(a,wks,u_zoom,v_zoom,opts)
        delete(opts)


      ; MAKE PLOTS                                       

        if ( pressure .eq. 800 ) then   
          opts_z@ContourParameters = (/ 5.0 /)
          contour_height = wrf_contour(a,wks,z_plane,opts_z)
          plot = wrf_map_overlays(a,wks,(/contour_qv,vector/),pltres,mpres)
        end if

        if ( pressure .eq. 700 ) then   
          opts_z@ContourParameters = (/ 5.0 /)
          contour_height = wrf_contour(a,wks, z_plane,opts_z)
          plot = wrf_map_overlays(a,wks,(/contour_qv,vector/),pltres,mpres)
        end if


        if ( pressure .eq. 600 ) then   
          opts_z@ContourParameters = (/ 5.0 /)
          contour_height = wrf_contour(a,wks,z_plane,opts_z)
          plot = wrf_map_overlays(a,wks,(/contour_qv,vector/),pltres,mpres)
        end if


        delete(opts_z)

    end do      ; END OF LEVEL LOOP

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  end do        ; END OF TIME LOOP

end
