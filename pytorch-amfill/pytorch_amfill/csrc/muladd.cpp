#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <vector>

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}



void amfill_cpu_common(const at::Tensor& im, const at::Tensor& mask, at::Tensor& om) {

  auto im_sizes = im.sizes();
  TORCH_CHECK(im_sizes == mask.sizes());
  TORCH_CHECK(im_sizes == om.sizes());
  TORCH_CHECK(im.dtype() == at::kFloat);
  TORCH_CHECK(om.dtype() == at::kFloat);
  TORCH_CHECK(mask.dtype() == at::kBool);
  TORCH_INTERNAL_ASSERT(im.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(om.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(mask.device().type() == at::DeviceType::CPU);
  auto im_accessor = im.accessor<float, 2>();
  auto om_accessor = om.accessor<float, 2>();
  auto mask_accessor = mask.accessor<bool, 2>();

  for (int64_t idy = 0; idy < im_sizes[0]  ; idy++)
    for (int64_t idx = 0; idx < im_sizes[1]  ; idx++) {
      if (mask_accessor[idy][idx])
        om_accessor[idy][idx] = im_accessor[idy][idx];
      else {

        // find nearest unmasked
        const int mcrad = std::max(im_sizes[1], im_sizes[0]) / 2;
        int crad=0; // negative crad indicates first nonmasked pixel
        while ( crad>=0 && crad < mcrad ) {
          crad++;
          const int cr2 = crad*crad;
          for (int dy = -crad ; dy <= crad ; dy++) {
            if ( idy < -dy || idy >= im_sizes[0]-dy )
              continue;
            const int dx = floor( sqrt((float)(cr2-dy*dy)) );
            if  ( ( idx-dx >= 0  &&  mask_accessor[idy][idx-dx] )
               || ( idx+dx < im_sizes[1]  &&  mask_accessor[idy][idx+dx] ) ) { // found
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
          if (idyy<0 || idyy>=im_sizes[0])
            continue;
          const int dy2 = dy*dy;
          const int xrad = floor(sqrt(2*sig22-dy2));
          for (int dx = -xrad ; dx <= xrad ; dx++) {
            const int idxx = idx+dx;
            if (idxx<0 || idxx>=im_sizes[1] || ! mask_accessor[idyy][idxx])
              continue;
            const float wght = exp(-(dx*dx + dy2)/sig22);
            mass += wght;
            gsumd += wght*im_accessor[idyy][idxx];
          }
        }
        if (mass>0.0)
          om_accessor[idy][idx] = gsumd / mass;

      }
    }

}

namespace pytorch_amfill {

at::Tensor amfill_cpu(const at::Tensor& im, const at::Tensor& mask) {
  at::Tensor om = at::empty(im.sizes(), im.options());
  amfill_cpu_common(im, mask, om);
  return om;
}


// An example of an operator that mutates one of its inputs.
at::Tensor amfill_cpu_(at::Tensor& iom, const at::Tensor& mask) {
  amfill_cpu_common(iom, mask, iom);
  return iom;
}

// Defines the operators
TORCH_LIBRARY(pytorch_amfill, m) {
  m.def("amfill(Tensor im, Tensor mask) -> Tensor");
  m.def("amfill_(Tensor im, Tensor mask) -> Tensor");
}

// Registers CUDA implementations
TORCH_LIBRARY_IMPL(pytorch_amfill, CPU, m) {
  m.impl("amfill", &amfill_cpu);
  m.impl("amfill_", &amfill_cpu_);
}

}
