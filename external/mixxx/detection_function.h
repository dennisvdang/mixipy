/* -*- c-basic-offset: 4 indent-tabs-mode: nil -*-  vi:set ts=8 sts=4 sw=4: */

/*
    QM DSP Library

    Centre for Digital Music, Queen Mary, University of London.
    This file 2005-2006 Christian Landone.

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation; either version 2 of the
    License, or (at your option) any later version.  See the file
    COPYING included with this distribution for more information.
*/

#ifndef MIXIPY_DETECTION_FUNCTION_H
#define MIXIPY_DETECTION_FUNCTION_H

#include <vector>
#include "math_utilities.h"
#include "math_aliases.h"
#include "phase_vocoder.h"
#include "window.h"

#define DF_HFC (1)
#define DF_SPECDIFF (2)
#define DF_PHASEDEV (3)
#define DF_COMPLEXSD (4)
#define DF_BROADBAND (5)

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

private:
    void whiten();
    double run_df();

    double hfc(int length, double* src);
    double spec_diff(int length, double* src);
    double phase_dev(int length, double* src_phase);
    double complex_sd(int length, double* src_magnitude, double* src_phase);
    double broadband(int length, double* src_magnitude);
        
    void initialize(DFConfig config);
    void deinitialize();

    int m_df_type;
    int m_data_length;
    int m_half_length;
    int m_step_size;
    double m_db_rise;
    bool m_whiten;
    double m_whiten_relax_coeff;
    double m_whiten_floor;

    double* m_mag_history;
    double* m_phase_history;
    double* m_phase_history_old;
    double* m_mag_peaks;

    double* m_windowed;
    double* m_magnitude;
    double* m_theta_angle;
    double* m_unwrapped;

    Window<double>* m_window;
    PhaseVocoder* m_phase_voc;
};

#endif  // MIXIPY_DETECTION_FUNCTION_H
