load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl" 
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRFUserARW.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRF_contributed.ncl"
begin
  ;; read data
  ;; systemfunc: Executes a shell command and returns the output.
  ;; Retrieve a list of files from one directory
  wrfout = systemfunc("ls ../../../../../sf_sfclay_phys\=1/wrfout_d02_2023-08-20_06:00:00") ;wrfout_d01_*")  
  filnum = dimsizes(wrfout) 
  data   = addfiles(wrfout, "r") 
  wrflat = data[0]->XLAT    
  wrflon = data[0]->XLONG
  dims   = dimsizes(wrflat)
  ;; get the dimension size of wrf
  nt     = dims(0)
  nlat   = dims(1)
  nlon   = dims(2)
;printVarSummary(wrflat)
  res=True
  res@gsnDraw=False               ;-- don't draw plot yet
  res@gsnFrame=False              ;-- don't advance frame
  res@gsnMaximize=True            ;-- don't maximized the workstation
  res@gsnLeftString=""
  res@gsnRightString=""

  res@pmTickMarkDisplayMode="Conditional"         ;-- displays TickMark object

  res@tmXTOn=False    ;-- turns off top tick marks
  res@tmYROn=False    ;-- turns off right tick marks

  res@tfDoNDCOverlay = True

	resspd	= res
	restream= res
	resp	= res
	resm	= res
        resrh   = res

	color = read_colormap_file("hawaii")   ; read the color map
	ncolors = dimsizes(color(:,0))
	nskip_beg = 0
	nskip_end = 20
	color = color(::-1,:)
	color_trim = color(nskip_beg:ncolors-1-nskip_end,:)

; We are interested in a zoomed area with the SW corner at 20N;90W,
; and the NE corner at 30N;80W
; So get the XY points of these points
  lats = (/  34.0,  40.0 /)
  lons = (/ -118.0, -112.0 /)
  loc = wrf_user_ll_to_ij(data[0], lons, lats, True)


; loc(0,;) is west-east (x) ; loc(1,:) is south-north (y)
; subtract one since we want to use it as an index in NCL
  x_start = loc(0,0) -1
  x_end   = loc(0,1) -1
  y_start = loc(1,0) -1
  y_end   = loc(1,1) -1



  res@cnLinesOn      = False
;  res@cnInfoLabelOn  = False               ;-- not draw an informational label
  res@cnLineLabelBackgroundColor = "Transparent"       ;-- draw lines background color
  res@cnLineThicknessF = 7                 ;-- thickness for all contour lines
  res@cnLineLabelPlacementMode = "constant"   ;-- draws line labels as an integral part of the line dash pattern
  res@cnLineLabelInterval  = 1   ;-- set label intervals on contours
  res@cnLineLabelDensityF  = 1   ;-- set number of labels along a contour line

  res@mpOutlineBoundarySets       = "GeophysicalAndUSStates"
  res@mpOutlineOn                 = True  ;-- outline land (default: False)
  res@mpGeophysicalLineThicknessF = 7
  res@mpNationalLineThicknessF    = 7
  res@mpUSStateLineThicknessF     = 7
  res@mpFillOn                    = False   ;True ;--map fill
  res@mpProjection           = "mercator"  ;
  res@mpUSStateLineColor     = "black"
  res@mpNationalLineColor    = "black"
  res@mpGeophysicalLineColor = "black"

  res@ZoomIn = True        ; set up map info for zoomed area
  res@Xstart = x_start
  res@Ystart = y_start
  res@Xend = x_end
  res@Yend = y_end
  res = wrf_map_resources(data[0],res)


 
  res@lbLabelBarOn        = True          ; turn off individual cb's


  resm@cnLevelSelectionMode =  "ManualLevels" ; Montgomery Streamfunction
  resm@cnMinLevelValF       = 310000.
  resm@cnMaxLevelValF       = 324000.
  resm@cnLevelSpacingF      = 100.
;  resm@cnFillPalette        = "wh-bl-gr-ye-re"
  resm@cnSpanFillPalette    = True
  resm@cnFillOn             = False         ; color plot desired
  resm@cnLinesOn            = True
  resm@cnLineColor          = "Black"
  resm@cnInfoLabelFontColor = "Black"
  resm@cnInfoLabelOrthogonalPosF = 0.25
  resm@cnLineLabelPlacementMode = "constant"   ;-- draws line labels as an integral part of the line dash pattern
  resm@cnLineLabelInterval  = 4   ;-- set label intervals on contours
  resm@cnLineLabelDensityF  = 1   ;-- set number of labels along a contour line
;  resm@cnLineLabelFontColor = "Transparent"       ;-- draw lines background color
  resm@cnLineLabelInterval  = 4   ;-- set label intervals on contours
  resm@cnLineThicknessF = 7


  res@cnLevelSelectionMode =  "ManualLevels" ; SPD
  res@cnMinLevelValF       = 10.
  res@cnMaxLevelValF       = 40.
  res@cnLevelSpacingF      = 2.
  res@cnFillPalette        = color_trim

  res@cnSpanFillPalette    = True
  res@cnFillOn             = True         ; color plot desired
;  res@cnLinesOn            = False
  res@cnLineColor          = "Blue"
  res@cnInfoLabelFontColor = "Blue"
  res@cnInfoLabelOrthogonalPosF = 0.25
  res@cnLineLabelPlacementMode = "constant"   ;-- draws line labels as an integral part of the line dash pattern
  res@cnLineLabelInterval  = 2   ;-- set label intervals on contours
  res@cnLineLabelDensityF  = 1   ;-- set number of labels along a contour line
  res@cnLineLabelFontColor = "Blue" ;Transparent"       ;-- draw lines background color
  res@cnLineThicknessF = 2 

  restream@stArrowLengthF     = 0.008                    ; size of the arrows.
  restream@stMinArrowSpacingF = 0.03 ;0.004                    ; arrow spacing.
;  restream@stArrowStride      = 100                        ; arrows start every third
  restream@stMinDistanceF     =  0.03 ;0.014 ;0.009
  restream@stMinArrowSpacingF = 0.03 
  restream@stLineThicknessF   = 1.1
  restream@cnInfoLabelOn      = True ;False
  restream@stLineThicknessF   = 2.0
  restream@stExplicitLabelBarLabelsOn = True


;type = "x11"
type = "png"

type@wkWidth = 3000
type@wkHeight = 3000


  do i = 0, filnum-1


    theta = wrf_user_getvar(data[i], "theta", -1)
    Time  = wrf_user_getvar(data[i], "times", -1) ;-- get variable : Times in file (return strings)
    u     = wrf_user_getvar(data[i], "ua", -1)
    v     = wrf_user_getvar(data[i], "va", -1)
    rh    = wrf_user_getvar(data[i], "rh", -1)
    p     = wrf_user_getvar(data[i], "pressure", -1)
    tk    = wrf_user_getvar(data[i], "tk", -1)
    z     = wrf_user_getvar(data[i], "z", -1)
printVarSummary(p)
 
	q = mixhum_ptrh(p, tk, rh, 2)
;	printVarSummary(q)

	SE = static_energy_atm(tk, q, z, (/1,0/))
	MSE = SE[0]   ; Moist Static Energy MSE = DSE +LSE (Units J/kg = m2/s2)
	DSE = SE[1]   ; Dry Static Energy  DSE = Cp*t + g*z (Montgomery Streamfunction) (Units J/kg = m2/s2)
	LSE = SE[2]   ; Latent Static Energy LSE =Lv*q (Units J/kg = m2/s2)

dse_min = min(DSE)
dse_max = max(DSE)
print(dse_min)
print(dse_max)
;	printVarSummary(DSE)



        U_isentropic = wrf_user_interp_level(u, theta, 316, False)
        V_isentropic = wrf_user_interp_level(v, theta, 316, False)
        RH_isentropic = wrf_user_interp_level(rh, theta, 316, False)
        P_isentropic = wrf_user_interp_level(p, theta, 316, False)
        DSE_isentropic = wrf_user_interp_level(DSE, theta, 316, False)

       spd_isentropic = sqrt(U_isentropic*U_isentropic + V_isentropic*V_isentropic)

    U_zoom = U_isentropic(y_start:y_end,x_start:x_end)
    V_zoom = V_isentropic(y_start:y_end,x_start:x_end)
    RH_zoom = RH_isentropic(y_start:y_end,x_start:x_end)
    P_zoom = P_isentropic(y_start:y_end,x_start:x_end)
    DSE_zoom = DSE_isentropic(y_start:y_end,x_start:x_end)
    printVarSummary(DSE_zoom)
        spd_zoom = sqrt(U_zoom^2 + V_zoom^2)
    wrf_smooth_2d(DSE_isentropic, 15)



    Times = data[i]->Times 
    if (i.eq.0) then
    timelabel0 = Times(0,0:12)
    end if

   do j = 0, dimsizes(Time)-1

    print(Time(j))
    timelabel = Times(j,5:12)

    wks=gsn_open_wks(type, "Isentropic_316K_SPD_PSI_D02"+Time(j))
;    wks=gsn_open_wks("png", "Isentropic_316K_SPD_PSI_D02_"+Time(j)) 
    res@gsnLeftString  = "316K SPD (Fill in m/s) & PSI (Contour in m**2/s**2) at " + timelabel

    plot1 = gsn_csm_contour_map(wks,spd_zoom(:,:),res)
    plot2 = gsn_csm_contour(wks,DSE_zoom(:,:),resm) ;Regular Streamline


	overlay(plot1,plot2)

; Add locations to the plot: Polymarker and Text.

  lats  = (/   36.05/)
  lons  = (/  -115.18/)
  tstrs = (/"VEF  "/)
  mkres = True
  txres = True
  mkres@gsMarkerIndex = 16 ;5 ; left_triangle_filled
  mkres@gsMarkerThicknessF = 3 ;6 ; thickness of the marker
  mkres@gsMarkerSizeF = 0.01 ; size of the marker
  mkres@gsMarkerOpacityF = 1.0
  mkres@gsMarkerColor = 1 ; black color of the marker
  txres@txFontHeightF = 0.011
  txres@txJust        = "CenterRight"

; Now put it on the plot

  mkid = gsn_add_polymarker(wks,plot1,lons,lats,mkres)
  txid = gsn_add_text(wks,plot1,"   " + tstrs,lons,lats,txres)

    draw(plot1)
    frame(wks)
   end do
  end do

end
