%module detection_function

%{
#include "../cpp/detection_function.h"
%}

%include "std_vector.i"
%include "std_string.i"

%template(DoubleVector) std::vector<double>;

struct DFConfig {
    int step_size;
    int frame_length;
    int df_type;
    double db_rise;
    bool adaptive_whitening;
    double whitening_relax_coeff;
    double whitening_floor;
};

class DetectionFunction {
public:
    DetectionFunction(DFConfig config);
    virtual ~DetectionFunction();

    double process_time_domain(const std::vector<double>& samples);
    std::vector<double> get_spectrum_magnitude();
};