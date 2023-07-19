@pyexec """
import ehtim as eh
import numpy as np

global np, pmodes, quick_analysis, eh

def pmodes(im, ms, r_min=0, r_max=1e3, norm_in_int = False, norm_with_StokesI = True, intensityRatioForAnalysis=None):

  if type(im) == eh.image.Image:
    npix = im.xdim
    iarr = im.ivec.reshape(npix, npix)
    qarr = im.qvec.reshape(npix, npix)
    uarr = im.uvec.reshape(npix, npix)
    varr = im.vvec.reshape(npix, npix)
    fov_muas = im.fovx()/eh.RADPERUAS

  else:
    hfp = h5py.File(im,'r')
    DX = hfp['header']['camera']['dx'][()]
    dsource = hfp['header']['dsource'][()]
    lunit = hfp['header']['units']['L_unit'][()]
    scale = hfp['header']['scale'][()]
    pol = np.flip(np.copy(hfp['pol']).transpose((1,0,2)),axis=0) * scale
    hfp.close()
    fov_muas = DX / dsource * lunit * 2.06265e11
    npix = pol.shape[0]
    iarr = pol[:,:,0]
    qarr = pol[:,:,1]
    uarr = pol[:,:,2]
    varr = pol[:,:,3]

  if intensityRatioForAnalysis is not None:
    mask = iarr < intensityRatioForAnalysis*np.max(iarr)
    iarr[mask] = 0
    qarr[mask] = 0
    uarr[mask] = 0
    varr[mask] = 0

  parr = qarr + 1j*uarr
  normparr = np.abs(parr)
  marr = parr/iarr
  phatarr = parr/normparr
  area = (r_max*r_max - r_min*r_min) * np.pi
  pxi = (np.arange(npix)-0.01)/npix-0.5
  pxj = np.arange(npix)/npix-0.5
  mui = pxi*fov_muas
  muj = pxj*fov_muas
  MUI,MUJ = np.meshgrid(mui,muj)
  MUDISTS = np.sqrt(np.power(MUI,2.)+np.power(MUJ,2.))

  # get angles measured East of North
  PXI,PXJ = np.meshgrid(pxi,pxj)
  angles = np.arctan2(-PXJ,PXI) - np.pi/2.
  angles[angles<0.] += 2.*np.pi

  # get flux in annulus
  tf = iarr [ (MUDISTS<=r_max) & (MUDISTS>=r_min) ].sum()

  # get total polarized flux in annulus
  pf = normparr [ (MUDISTS<=r_max) & (MUDISTS>=r_min) ].sum()

  #get number of pixels in annulus
  npix = iarr [ (MUDISTS<=r_max) & (MUDISTS>=r_min) ].size

  #get number of pixels in annulus with flux >= some % of the peak flux
  ann_iarr = iarr [ (MUDISTS<=r_max) & (MUDISTS>=r_min) ]
  peak = np.max(ann_iarr)
  num_above5 = ann_iarr[ann_iarr > .05* peak].size
  num_above10 = ann_iarr[ann_iarr > .1* peak].size

  # compute betas
  betas = []
  for m in ms:
    qbasis = np.cos(-angles*m)
    ubasis = np.sin(-angles*m)
    pbasis = qbasis + 1.j*ubasis
    if norm_in_int:
      if norm_with_StokesI:
        prod = marr * pbasis
      else:
        prod = phatarr * pbasis
      coeff = prod[ (MUDISTS <= r_max) & (MUDISTS >= r_min) ].sum()
      coeff /= npix
    else:
      prod = parr * pbasis
      coeff = prod[ (MUDISTS<=r_max) & (MUDISTS>=r_min) ].sum()
      if norm_with_StokesI:
        coeff /= tf
      else:
        coeff /= pf
    betas.append(coeff)


  return betas


def quick_analysis(im, beta_ms=range(6), verbose=False, resolution_muas=0.0, intensityRatioForAnalysis=None):

	#Compute beta modes.
	betas = pmodes(im.blur_circ(resolution_muas*eh.RADPERUAS,fwhm_pol=resolution_muas*eh.RADPERUAS), beta_ms, intensityRatioForAnalysis=intensityRatioForAnalysis)

	#Assembly Stokes arrays.
	npix = im.xdim
	iarr = im.ivec.reshape(npix, npix)
	qarr = im.qvec.reshape(npix, npix)
	uarr = im.uvec.reshape(npix, npix)
	varr = im.vvec.reshape(npix, npix)

	#We may want to consider masking the image in some areas.
	if intensityRatioForAnalysis is not None:
		mask = iarr < intensityRatioForAnalysis * np.max(iarr)
		iarr[mask] = 0
		qarr[mask] = 0
		uarr[mask] = 0
		varr[mask] = 0

	#Net polarization.
	m_net = np.sqrt(np.sum(qarr)**2 + np.sum(uarr)**2) / np.sum(iarr)
	v_net = np.sum(varr) / np.sum(iarr)

	#Blur and obtain average polarization.
	im_blurred = im.blur_circ(resolution_muas*eh.RADPERUAS,fwhm_pol=resolution_muas*eh.RADPERUAS)
	iarr_blurred = im_blurred.ivec.reshape(npix, npix)
	qarr_blurred = im_blurred.qvec.reshape(npix, npix)
	uarr_blurred = im_blurred.uvec.reshape(npix, npix)
	parr_blurred = np.sqrt(qarr_blurred**2 + uarr_blurred**2)
	varr_blurred = im_blurred.vvec.reshape(npix, npix)
	m_avg = np.sum(parr_blurred) / np.sum(iarr_blurred)
	v_avg = np.sum(np.abs(varr_blurred)) / np.sum(iarr_blurred)

	#Print what you found, if desired.
	if verbose:
		print("Beta Modes:")
		for i in beta_ms:
			print(f"   {i}: |beta_{i}|={np.abs(betas[i]):1.3f}, arg(beta_{i})={np.angle(betas[i], deg=True):1.3f}")
		print(f"|beta_2|/|beta_1| = {np.abs(betas[2])/np.abs(betas[1]):1.3f}")
		print(f"|beta_2|/sum(|beta_i|) = {np.abs(betas[2])/np.sum(np.abs(betas)):1.3f}")
		print(f"m_net = {m_net:1.3f}")
		print(f"m_avg = {m_avg:1.3f}")
		print(f"v_net = {v_net:1.3f}")
		print(f"v_avg = {v_avg:1.3f}")

	#Output a dictionary.
	D = {}
	D['m_net'] = m_net
	D['m_avg'] = m_avg
	D['v_net'] = v_net
	D['v_avg'] = v_avg
	D['beta_ms'] = beta_ms
	D['betas'] = betas
	D['resolution_muas'] = resolution_muas

	return D

"""
quick_analysis(x) = @pyeval("quick_analysis")(x)
