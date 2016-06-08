#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/repeat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RepeatLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const int num_output = this->layer_param_.repeat_param().num_repeats();
  N_ = num_output;

  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.repeat_param().axis());
  K_ = bottom[0]->count(axis);

  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    vector<int> weight_shape(2);
    weight_shape[0] = N_ * K_;
    weight_shape[1] = K_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    Dtype* weight_data = this->blobs_[0]->mutable_cpu_data();
    for( int i = 0; i < N_  ; i ++  )
      for( int j = 0 ; j < K_ ; j ++ )
       caffe_set(1, Dtype(1), weight_data + this->blobs_[0]->offset(  j + i * K_,  j ) );
  }
}

template <typename Dtype>
void RepeatLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.repeat_param().axis());
  M_ = bottom[0]->count(0, axis);

  vector<int> bottom_shape = bottom[0]->shape();
  vector<int> top_shape ;
  top_shape.push_back( bottom_shape[0] );
  top_shape.push_back( N_ );
  for( int i = 1; i < bottom_shape.size() ; i ++ )
    top_shape.push_back( bottom_shape[i]);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void RepeatLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const Dtype* weight = this->blobs_[0]->cpu_data();
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
        M_, N_ * K_, K_, (Dtype)1.,
        bottom_data, weight, (Dtype)0., top_data);
}

template <typename Dtype>
void RepeatLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_ * K_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(RepeatLayer);
#endif

INSTANTIATE_CLASS(RepeatLayer);
REGISTER_LAYER_CLASS(Repeat);

}  // namespace caffe
