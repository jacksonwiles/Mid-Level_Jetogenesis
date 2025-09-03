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
  wrfout = systemfunc("ls ../../../../sf_sfclay_phys\=1/wrfout_d02_2023-08-*")  
  filnum = dimsizes(wrfout) 
  data   = addfiles(wrfout, "r") 
  wrflat = data[0]->XLAT    
  wrflon = data[0]->XLONG
  dims   = dimsizes(wrflat)
  ;; get the dimension size of wrf
  nt     = dims(0)
  nlat   = dims(1)
  nlon   = dims(2)

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

  res@mpOutlineBoundarySets       = "GeophysicalAndUSStates"
  res@mpOutlineOn                 = True  ;-- outline land (default: False)
  res@mpGeophysicalLineThicknessF = 1
  res@mpNationalLineThicknessF    = 1
  res@mpFillOn                    = False   ;True ;--map fill
  res@mpProjection           = "mercator"  ;
;  res@mpLambertParallel1F    = 28.406
;  res@mpLambertParallel2F    = 0.
;  res@mpLambertMeridianF     = -116.095
  res@mpUSStateLineColor     = "black"
  res@mpNationalLineColor    = "black"
  res@mpLimitMode            = "Corners"
  res@mpLeftCornerLatF = wrflat(0,5,5)              ;-- min latitude
  res@mpLeftCornerLonF = wrflon(0,5,5)              ;-- min longitude
  res@mpRightCornerLatF = wrflat(0,nlat-5,nlon-5)   ;-- max latitude
  res@mpRightCornerLonF = wrflon(0,nlat-5,nlon-5)   ;-- max longitude
;  res@mpOutlineBoundarySets  = "AllBoundaries"
 
  res@lbLabelBarOn        = True          ; turn off individual cb's

  res@cnLevelSelectionMode =  "ManualLevels" ;
  res@cnMinLevelValF       = 0.
  res@cnMaxLevelValF       = 50.
  res@cnLevelSpacingF      = 5.
  res@cnFillPalette        = "wh-bl-gr-ye-re"
  res@cnSpanFillPalette    = True
  res@cnFillOn             = True         ; color plot desired

  do i = 0, filnum-1
;    reflect = wrf_user_getvar(data[i], "dbz", -1)
    mreflect = wrf_user_getvar(data[i], "mdbz", -1)
;    wrfdbz = reflect+mreflect 
    Time  = wrf_user_getvar(data[i], "times", -1) ;-- get variable : Times in file (return strings)

    Times = data[i]->Times 
    if (i.eq.0) then
    timelabel0 = Times(0,0:12)
    end if

   do j = 0, dimsizes(Time)-1

    print(Time(j))
    timelabel = Times(j,5:12)

;    wks=gsn_open_wks("x11", "Reflectivity"+Time(j))
    wks=gsn_open_wks("png", "Reflectivity_D02_"+Time(j)) 
    res@gsnLeftString  = "WRF-ARW Reflectivity at " + timelabel
;    res@gsnRightString  = timelabel
    plot1 = gsn_csm_contour_map(wks,mreflect(j,5:nlat-5,5:nlon-5),res)
    draw(plot1)
    frame(wks)
   end do
;    delete(reflect)
    delete(mreflect)
 ;   delete(wrfdbz)
    delete(Time)
    delete(Times)
    delete(plot1)
  end do

end
