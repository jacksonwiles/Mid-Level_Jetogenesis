
;   Example script to produce plots for a WRF real-data run,
;   with the ARW coordinate dynamics option.
;   Plot data on a cross section
;   This script will plot data from a a given point A to point B
;   Vertical coordinate is pressure

load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRFUserARW.ncl"

begin
;
; The WRF ARW input file.  
; This needs to have a ".nc" appended, so just do it.
;  a = addfile("../../../../sf_sfclay_phys=1/wrfout_d02_2023-08-19_23:00:00" + ".nc", "r")

wrfout	= systemfunc("ls ../../../../sf_sfclay_phys\=1/wrfout_d02_2023-08-20_00:00:00") ;../../../../../sf_sfclay_phys=1/wrfout_d02_*")
filnum	= dimsizes(wrfout)
a	= addfiles(wrfout, "r")

wrflat = wrf_user_getvar(a,"XLAT",-1)
wrflon = wrf_user_getvar(a,"XLONG",-1)
dimms   = dimsizes(wrflat)
  nt     = dimms(0)
  nlat   = dimms(1) 
  nlon   = dimms(2)
 

; We generate plots, but what kind do we prefer?
; type = "x11"
;  type = "pdf"
  type = "png"
; type = "ncgm"
type@wkWidth = 3000
type@wkHeight = 3000



; Set some basic resources
  res = True
  res@MainTitle = "OAK - SLC"
  res@vpWidthF = .9               ; overwrite basic plot size
  res@vpHeightF = 1.0
  res@Footer = False

  res@tmXBLabelFontHeightF = 0.035
  res@tmYLLabelFontHeightF = 0.06

        color = read_colormap_file("hawaii")   ; read the color map
        ncolors = dimsizes(color(:,0))
        nskip_beg = 0
        nskip_end = 20
        color = color(::-1,:)
        color_trim = color(nskip_beg:ncolors-1-nskip_end,:)



  pltres = True

do k = 0, filnum-1
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  FirstTime = True
  times  = wrf_user_getvar(a[k],"times",-1) ; get times in the file
  ntimes = dimsizes(times)         ; number of times in the file

  mdims = getfilevardimsizes(a[k],"P") ; get some dimension sizes for the file
  nd = dimsizes(mdims)

;---------------------------------------------------------------

  do it = 0,ntimes-1               ; TIME LOOP

    print("Working on time: " + times(it) )
    res@TimeLabel = times(it)   ; Set Valid time to use on plots

  wks = gsn_open_wks(type,"D02_X_Sect_SPD_Vector" + times(it))

    th = wrf_user_getvar(a[k],"th",it)        ; theta
    u = wrf_user_getvar(a[k],"ua",it)
    v = wrf_user_getvar(a[k],"va",it)
    w = wrf_user_getvar(a[k],"wa",it)
    p  = wrf_user_getvar(a[k], "pressure",it) ; grid point height

  lats = wrf_user_getvar(a[k], "lat",0)
  lons = wrf_user_getvar(a[k], "lon",0)


	start_lat = 37.73
	start_lon = -122.21
	end_lat = 40.77
	end_lon = -111.95

    latlon = (/start_lon,start_lat,end_lon,end_lat/)
    opts             = True
    opts@latlon      = True
    opts@linecoords  = True
    opts@file_handle = a[k]

    rh_latlon       = wrf_user_vert_cross(th,p,latlon,opts)





	loc1  = wrf_user_ll_to_ij(a[k], start_lon, start_lat, res)
;        print("X/Y locations are: " + loc1)
;        print(loc1(0))
;	print(loc1(1))

        loc2  = wrf_user_ll_to_ij(a[k], end_lon, end_lat, res)
;        print("X/Y locations are: " + loc2)
;        print(loc2(0))
;        print(loc2(1))


        plane = new(4,float)
        plane = (/  loc1(0),loc1(1) ,  loc2(0),loc2(1)  /) ; start x;y & end x;y point

        th_plane = wrf_user_intrp3d(th,p,"v",plane,0.,opts)
        u_plane  = wrf_user_intrp3d(u,p,"v",plane,0.,opts)
        v_plane  = wrf_user_intrp3d(v,p,"v",plane,0.,opts)
        w_plane  = wrf_user_intrp3d(w,p,"v",plane,0.,opts)
  

        spd = (u_plane^2 + v_plane^2 + w_amplify^2)^0.5 ; m/sec
        spd@description = "Wind Speed" ;; w amplified by 20"
        spd@units = "~N~ms~S~-1~N~"

 
      ; Let's create nice labels - only have to do this once
        if ( FirstTime ) then
          zmax = 51.     ;  Place top at model top or close to zmax
          zz = wrf_user_intrp3d(p,p,"v",plane,0.,opts)
          zmin = 1000
          nice_levs = floor((zmin-zmax)/50)*50
          zmax = zmin - nice_levs
          dims = dimsizes(zz)
          zmax_pos = dims(0)-1
          do imax = 1,dims(0)-1
            if ( .not.ismissing(zz(imax,0)) .and. zz(imax,0) .ge. zmax ) then
              zmax_pos = imax
            end if
          end do
          zspan = zmax_pos
          zmax = zz(zmax_pos,0)
          nz = floattoint((zmin-zmax)/50+1)
          FirstTime = False
        end if

        

      ; Options for XY Plots
        opts_xy                         = res
        opts_xy@tiYAxisString           = "Pressure (hPa)"
        opts_xy@tiXAxisString           = "Latitude & Longitude"
        opts_xy@AspectRatio             = 0.75
        opts_xy@cnMissingValPerimOn     = True
        opts_xy@cnMissingValFillColor   = "sienna4"
        opts_xy@cnMissingValFillPattern = "SolidFill"
        opts_xy@tmYLMode                = "Explicit"
        opts_xy@tmYLValues              = fspan(0,zspan,nz)                    ; Create tick marks
        opts_xy@tmYLLabels              = sprintf("%.0f",fspan(zmin,zmax,nz))  ; Create labels
        opts_xy@tiXAxisFontHeightF      = 0.020
        opts_xy@tiYAxisFontHeightF      = 0.010
        opts_xy@tmXBMajorLengthF        = 0.02
        opts_xy@tmYLMajorLengthF        = 0.02
        opts_xy@tmYLLabelFontHeightF    = 0.0001
        opts_xy@PlotOrientation         = th_plane@Orientation
        opts_xy@tiYAxisFontHeightF      = 0.03
        ;opts_xy@lgLabelFontHeightF  = 1.5
        ;opts_xy@units = "~N~ms~S~-1~N~"

        ;--Explicitly set lat/lon labels for X axis
        xvalues = ispan(0,dimsizes(rh_latlon(0,:))-1,1)
        ll_step = 70                                       ; step interval for tickmarks
        opts_xy@tmXBMode          = "Explicit"
        opts_xy@tmXBValues        = xvalues(::ll_step)
        opts_xy@tmXBLabels        = sprintf("%6.1f",rh_latlon@lats(::ll_step)) + "~S~o~N~N~C~" + \
                          sprintf("%6.1f",rh_latlon@lons(::ll_step)) + "~S~o~N~W"

	opts_xy@tmXBMajorOutwardLengthF = 0.02
	opts_xy@tmXBMinorOutwardLengthF = 0.01
        opts_xy@tmYLMajorOutwardLengthF = 0.02
        opts_xy@tmYLMinorOutwardLengthF = 0.01
	opts_xy@tmYROn	=	False
	opts_xy@tmXTOn	=	False



       ; Plotting options for Theta
        opts_th = opts_xy
        opts_th@cnInfoLabelOrthogonalPosF = 0.00
        opts_th@gsnContourLineThicknessesScale = 2.0
        opts_th@ContourParameters  = (/ 290., 350., 2. /)
       ; opts_th@FieldTitle            = ""
        opts_th@cnInfoLabelOn = False


        ; Plotting options for Total Wind (U, V and W)
        opts_spd = opts_xy
        opts_spd@cnSpanFillPalette = True
        opts_spd@cnFillOn          = True
	opts_spd@cnLinesOn	   = False
	opts_spd@cnLineLabelInterval = 2
        opts_spd@gsnSpreadColor    = True
	opts_spd@gsnContourPosLineDashPattern = 1
	opts_spd@gsnContourLineThicknessesScale = 3.0
	opts_spd@cnLineLabelPlacementMode = "constant"   ;-- draws line labels as an integral part of the line dash pattern
	opts_spd@cnLineLabelDensityF  = 2   ;-- set number of labels along a contour line
        opts_spd@cnLineColor          = "Blue"
        opts_spd@cnLineLabelFontColor = "Blue"
        opts_spd@ContourParameters       = (/ 10., 50., 4. /)
        opts_spd@pmLabelBarOrthogonalPosF = -0.1
;        opts_spd@cnFillPalette         = "WhiteBlueGreenYellowRed" ;"BlAqGrYeOrReVi200"
	opts_spd@cnFillPalette		= color_trim
        ;opts_spd@vcLevelPalette = "NCV_bright"   

        ; Plotting options for vertical velocity
        opts_w = opts_xy
        opts_w@cnLinesOn          = False
        opts_w@ContourParameters  = (/ -2., 2., .4/)
        opts_w@cnFillOn           = True

        ; Plotting options for Wind Vector

        opts = opts_xy
        opts@FieldTitle = "Wind vector"   ; overwrite Field Title
       ; opts@vcRefAnnoOrthogonalPosF = -0.12
       ; opts@vcRefAnnoJust = "BottomRight"
       ; opts@vcRefLengthF = 0.03
       ; opts@vcWindBarbLineThicknessF = 6.0
        opts@NumVectors = 40       ; wind barb density
        opts@vcRefAnnoOn = True    ; define reference vector
        opts@vcRefMagnitudeF  = 20.0             ; define vector ref mag
        opts@vcMinDistanceF            = 0.04              ; Min distance between vectors
        opts@vcGlyphStyle              = "LineArrow"
        opts@vcLineArrowThicknessF = 1.5
        opts@vcLineArrowColor = "Black"





       ; Get the contour info  
	contour_th = wrf_contour(a[k],wks,th_plane(0:zmax_pos,:),opts_th)
        contour_spd = wrf_contour(a[k],wks,spd(0:zmax_pos,:),opts_spd)
        vector = wrf_vector(a[k],wks,u_plane,w_plane,opts)


       ; MAKE PLOTS         
        plot = wrf_overlays(a[k],wks,(/vector,contour_spd/),pltres)

       ; Delete options and fields, so we don't have carry over
        delete(u_plane)
        delete(v_plane)
        delete(w_plane)
        delete(spd)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  end do        ; END OF TIME LOOP
 end do
end
