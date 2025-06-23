#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace pytorch_amfill {


__device__ void amfill_root(const int  Xs, const int  Ys, const int idi,
                 const float* im, const bool* mask, float* om) {

  //const int idi = blockIdx.x * blockDim.x + threadIdx.x;
  const int idx = idi % Xs;
  const int idy = idi / Xs;
  if ( idx >= Xs || idy >= Ys )
    return;

  if (mask[idi]) {
    if (im != om)
      om[idi] = im[idi];
    return;
  }

  // find nearest unmasked
  const int mcrad = floor(sqrt(Xs*Xs + Ys*Ys))-1;
  int crad=0; // negative crad indicates first nonmasked pixel
  while ( crad>=0 && crad < mcrad ) {
    crad++;
    const int cr2 = crad*crad;
    for (int dy = -crad ; dy <= crad ; dy++) {
      const int idyy = idy+dy;
      if (idyy<0 || idyy>=Ys)
        continue;
      const int dx = floor( sqrt((float)(cr2-dy*dy)) );
      const int cshft = idyy * Xs + idx;
      if  ( ( idx-dx >= 0   &&  mask[cshft-dx] )
         || ( idx+dx < Xs  &&  mask[cshft+dx] ) ) { // found
        crad *= -1; // negative indicator set
        break;
      }
    }
  }
  if (crad>=0) // may happen only on full mask?
    return;
  const float sigma = -crad ; // final sigma, also negates previous indicator.
  const float sig22 = 2*sigma*sigma;
  const int mrad = floor(2.0*sigma);

  // filter
  float mass=0.0;
  float gsumd=0;
  for (int dy = -mrad ; dy <= mrad ; dy++) {
    const int idyy = idy+dy;
    if (idyy<0 || idyy>=Ys)
      continue;
    const int idiy = idyy * Xs;
    const int dy2 = dy*dy;
    const int xrad = floor(sqrt(2*sig22-dy2));
    for (int dx = -xrad ; dx <= xrad ; dx++) {
      const int idxx = idx+dx;
      const int idii = idiy + idxx;
      if (idxx<0 || idxx>=Xs || ! mask[idii] )
        continue;
      const float wght = exp(-(dx*dx + dy2)/sig22);
      mass += wght;
      gsumd += wght*im[idii];
    }
  }
  om[idi] = mass>0.0 ? gsumd / mass : 0;

}



__global__ void amfill_kernel(const int  Xs, const int  Ys,
                              const float* im, const bool* mask, float* om) {
  const int idi = blockIdx.x * blockDim.x + threadIdx.x;
  amfill_root(Xs, Ys, idi, im, mask, om);
}


__global__ void amfill_kernel_batch(const int Xs, const int Ys, const int Cs, const int Bs,
                                    const int MCs, const int MBs,
                                    const float* im, const bool* mask, float* om) {

  const int idi = blockIdx.x * blockDim.x + threadIdx.x;
  int rest = idi;
  const int idb = rest / (Xs*Ys*Cs);
  rest -= idb * (Xs*Ys*Cs);
  const int idc = rest / (Xs*Ys);
  rest -= idc * (Xs*Ys);
  const int idy = rest / Xs;
  const int idx = rest - idy * Xs;
  //const int idx = rest % Xs;
  //rest -= idx;
  //const int idy = rest % (Xs*Ys);
  //rest -= idy * Xs;
  //const int idc = rest % (Xs*Ys*Cs);
  //const int idb = rest / (Xs*Ys*Cs);
  if ( idx >= Xs || idy >= Ys || idc >= Cs || idb >= Bs )
    return;

  const int iShift = idb * Xs*Ys*Cs + idc * Xs*Ys;
  const int fShift = min(idb,MBs-1) * Xs*Ys*MCs + min(idc,MCs-1) * Xs*Ys;
  amfill_root(Xs, Ys, idx + idy * Xs , im+iShift, mask+fShift, om+iShift);

}



at::Tensor amfill_cuda(const at::Tensor& im, const at::Tensor& mask) {
  const int dim = im.dim();
  TORCH_CHECK(dim <= 4 and dim >= 2);
  const int imsizes[] = {
    dim == 4 ? std::max<int>(1,im.sizes()[0]) : 1,
    dim >= 3 ? std::max<int>(1,im.sizes()[dim-3]) : 1,
    im.sizes()[dim-2],
    im.sizes()[dim-1]
  };
  const int mdim = mask.dim();
  TORCH_CHECK(mdim <= 4 and mdim >= 2);
  const int mssizes[] = {
    mdim == 4 ? std::max<int>(1,mask.sizes()[0]) : 1 ,
    mdim >= 3 ? std::max<int>(1,mask.sizes()[mdim-3]) : 1 ,
    mask.sizes()[mdim-2],
    mask.sizes()[mdim-1]
  };
  TORCH_CHECK(imsizes[0] == mssizes[0]  or  mssizes[0] == 1);
  TORCH_CHECK(imsizes[1] == mssizes[1]  or  mssizes[1] == 1);
  TORCH_CHECK(imsizes[2] == mssizes[2]);
  TORCH_CHECK(imsizes[3] == mssizes[3]);
  TORCH_CHECK(im.dtype() == at::kFloat);
  TORCH_CHECK(im.dtype() == at::kFloat);
  TORCH_CHECK(mask.dtype() == at::kBool);
  TORCH_INTERNAL_ASSERT(im.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(mask.device().type() == at::DeviceType::CUDA);
  at::Tensor im_contig = im.contiguous();
  at::Tensor mask_contig = mask.contiguous();
  at::Tensor om = at::empty(im_contig.sizes(), im_contig.options());
  const float* im_ptr = im_contig.data_ptr<float>();
  const bool* mask_ptr = mask_contig.data_ptr<bool>();
  float* om_ptr = om.data_ptr<float>();
  int numel = im_contig.numel();
  amfill_kernel_batch<<<(numel+255)/256, 256>>>(imsizes[3], imsizes[2], imsizes[1], imsizes[0],
                                          mssizes[1], mssizes[0], im_ptr, mask_ptr, om_ptr);
  return om;
}




at::Tensor amfill_cuda_(at::Tensor& iom, const at::Tensor& mask) {
  const int dim = iom.dim();
  TORCH_CHECK(dim <= 4 and dim >= 2);
  const int imsizes[] = {
    dim == 4 ? std::max<int>(1,iom.sizes()[0]) : 1,
    dim >= 3 ? std::max<int>(1,iom.sizes()[dim-3]) : 1,
    iom.sizes()[dim-2],
    iom.sizes()[dim-1],
  };
  const int mdim = mask.dim();
  TORCH_CHECK(mdim <= 4 and mdim >= 2);
  const int mssizes[] = {
    mdim == 4 ? std::max<int>(1,mask.sizes()[0]) : 1 ,
    mdim >= 3 ? std::max<int>(1,mask.sizes()[mdim-3]) : 1 ,
    mask.sizes()[mdim-2],
    mask.sizes()[mdim-1],
  };
  TORCH_CHECK(imsizes[0] == mssizes[0]  or  mssizes[0] == 1);
  TORCH_CHECK(imsizes[1] == mssizes[1]  or  mssizes[1] == 1);
  TORCH_CHECK(imsizes[2] == mssizes[2]);
  TORCH_CHECK(imsizes[3] == mssizes[3]);
  TORCH_CHECK(iom.dtype() == at::kFloat);
  TORCH_CHECK(iom.dtype() == at::kFloat);
  TORCH_CHECK(mask.dtype() == at::kBool);
  TORCH_INTERNAL_ASSERT(iom.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(mask.device().type() == at::DeviceType::CUDA);
  at::Tensor iom_contig = iom.contiguous();
  at::Tensor mask_contig = mask.contiguous();
  float* iom_ptr = iom_contig.data_ptr<float>();
  const bool* mask_ptr = mask_contig.data_ptr<bool>();
  int numel = iom_contig.numel();
  amfill_kernel_batch<<<(numel+255)/256, 256>>>(imsizes[3], imsizes[2], imsizes[1], imsizes[0],
                                          mssizes[1], mssizes[0], iom_ptr, mask_ptr, iom_ptr);
  return iom;
}


// Registers CUDA implementations
TORCH_LIBRARY_IMPL(pytorch_amfill, CUDA, m) {
  m.impl("amfill", &amfill_cuda);
  m.impl("amfill_", &amfill_cuda_);
}

}

