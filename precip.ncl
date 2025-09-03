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
  wrfout = systemfunc("ls ./wrfout_precip/wrfout_d02_2023-08-*")  
  filnum = dimsizes(wrfout) 
  data   = addfiles(wrfout, "r") 
  wrflat = data[0]->XLAT    
  wrflon = data[0]->XLONG
  dims   = dimsizes(wrflat)
  ;; get the dimension size of wrf
  nt     = dims(0)
  nlat   = dims(1)
  nlon   = dims(2)

  wrfout_first_time = systemfunc("ls ./wrfout_precip/wrfout_d02_2023-08-19_21:00:00")
;  filnum = dimsizes(wrfout_first_time)
  a   = addfile(wrfout_first_time, "r")


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

  res@cnLinesOn      = False
  res@cnInfoLabelOn  = False               ;-- not draw an informational label
  res@cnLineLabelBackgroundColor = "Transparent"       ;-- draw lines background color
  res@cnLineThicknessF = 3                 ;-- thickness for all contour lines
  res@cnLineLabelPlacementMode = "constant"   ;-- draws line labels as an integral part of the line dash pattern
  res@cnLineLabelInterval  = 3   ;-- set label intervals on contours
  res@cnLineLabelDensityF  = 1   ;-- set number of labels along a contour line


; We are interested in a zoomed area with the SW corner at 20N;90W,
; and the NE corner at 30N;80W
; So get the XY points of these points
  lats = (/  34.0,  42.0 /)
  lons = (/ -122.0, -113.0 /)
  loc = wrf_user_ll_to_ij(data[0], lons, lats, True)


; loc(0,;) is west-east (x) ; loc(1,:) is south-north (y)
; subtract one since we want to use it as an index in NCL
  x_start = loc(0,0) -1
  x_end   = loc(0,1) -1
  y_start = loc(1,0) -1
  y_end   = loc(1,1) -1


  res@mpOutlineBoundarySets       = "GeophysicalAndUSStates"
  res@mpOutlineOn                 = True  ;-- outline land (default: False)
  res@mpGeophysicalLineThicknessF = 4
  res@mpNationalLineThicknessF    = 4
  res@mpUSStateLineThicknessF     = 4
  res@mpFillOn                    = False   ;True ;--map fill
 ; res@mpProjection           = "LambertConformal"  ;
 ; res@mpLambertParallel1F    = 36.101
 ; res@mpLambertParallel2F    = 36.101
 ; res@mpLambertMeridianF     = -80.711
  res@mpUSStateLineColor     = "black"
  res@mpNationalLineColor    = "black"
  res@mpGeophysicalLineColor = "black"
;  res@mpLimitMode            = "Corners"
;  res@mpLeftCornerLatF = wrflat(0,5,5)              ;-- min latitude
;  res@mpLeftCornerLonF = wrflon(0,5,5)              ;-- min longitude
;  res@mpRightCornerLatF = wrflat(0,nlat-5,nlon-5)   ;-- max latitude
;  res@mpRightCornerLonF = wrflon(0,nlat-5,nlon-5)   ;-- max longitude

  res@ZoomIn = True        ; set up map info for zoomed area
  res@Xstart = x_start
  res@Ystart = y_start
  res@Xend = x_end
  res@Yend = y_end
  res = wrf_map_resources(data[0],res)

;  res@mpOutlineBoundarySets  = "AllBoundaries" 
 
  res@lbLabelBarOn        = True          ; turn off individual cb's

  res@cnLevelSelectionMode =  "ManualLevels" ;
  res@cnMinLevelValF       = 1.
  res@cnMaxLevelValF       = 10.
  res@cnLevelSpacingF      = 0.5
  res@cnFillPalette        = "prcp_1" ;wh-bl-gr-ye-re" ;BlGrYeOrReVi200"
  res@cnSpanFillPalette    = True
  res@cnFillOn             = True         ; color plot desired

  rainc_first = wrf_user_getvar(a, "RAINC", -1)
  rainnc_first = wrf_user_getvar(a, "RAINNC", -1)
;  prcp_first = (rainc_first+rainnc_first) / 25.4

	rainc_first_zoom = rainc_first(:,y_start:y_end,x_start:x_end)
	rainnc_first_zoom = rainnc_first(:,y_start:y_end,x_start:x_end)

	prcp_first = (rainc_first_zoom+rainnc_first_zoom) / 25.4

  do i = 0, filnum-1
    rainc = wrf_user_getvar(data[i], "RAINC", -1)
    rainnc = wrf_user_getvar(data[i], "RAINNC", -1)
;    prcp = (rainc+rainnc) / 25.4 
;    printVarSummary(prcp)

	rainc_zoom = rainc(:,y_start:y_end,x_start:x_end)
	rainnc_zoom = rainnc(:,y_start:y_end,x_start:x_end)

	prcp = (rainc_zoom+rainnc_zoom) / 25.4
 
    Time  = wrf_user_getvar(data[i], "times", -1) ;-- get variable : Times in file (return strings)

    Times = data[i]->Times 
    if (i.eq.0) then
    timelabel0 = Times(0,5:12)
    end if

    do j = 0, dimsizes(Time)-1
  
   prcpt=prcp(j,:,:) - prcp_first(0,:,:)

    print(Time(j))
    timelabel = Times(j,5:12)

;    wks=gsn_open_wks("x11", "prcp"+Time(j))
    wks=gsn_open_wks("png", "accum_precip_"+Time(j)) 
    res@gsnLeftString  = "D02 Accum. Precip.(in) from " + timelabel0 + " to " + timelabel
;    res@gsnRightString  = timelabel
    plot1 = gsn_csm_contour_map(wks,prcpt(:,:),res)
    draw(plot1)
    frame(wks)
   end do
    delete(prcp)
    delete(prcpt)
    delete(rainc)
    delete(rainnc)
    delete(Time)
    delete(Times)
    delete(plot1)
  end do

end
