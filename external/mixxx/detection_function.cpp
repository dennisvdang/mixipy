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

#include "detection_function.h"
#include <cstring>
#include <math.h>

DetectionFunction::DetectionFunction(DFConfig config) : m_window(nullptr) {
    m_mag_history = nullptr;
    m_phase_history = nullptr;
    m_phase_history_old = nullptr;
    m_mag_peaks = nullptr;

    initialize(config);
}

DetectionFunction::~DetectionFunction() {
    deinitialize();
}

void DetectionFunction::initialize(DFConfig config) {
    m_data_length = config.frame_length;
    m_half_length = m_data_length / 2 + 1;

    m_df_type = config.df_type;
    m_step_size = config.step_size;
    m_db_rise = config.db_rise;

    m_whiten = config.adaptive_whitening;
    m_whiten_relax_coeff = config.whitening_relax_coeff;
    m_whiten_floor = config.whitening_floor;
    if (m_whiten_relax_coeff < 0) m_whiten_relax_coeff = 0.9997;
    if (m_whiten_floor < 0) m_whiten_floor = 0.01;

    m_mag_history = new double[m_half_length];
    std::memset(m_mag_history, 0, m_half_length * sizeof(double));

    m_phase_history = new double[m_half_length];
    std::memset(m_phase_history, 0, m_half_length * sizeof(double));

    m_phase_history_old = new double[m_half_length];
    std::memset(m_phase_history_old, 0, m_half_length * sizeof(double));

    m_mag_peaks = new double[m_half_length];
    std::memset(m_mag_peaks, 0, m_half_length * sizeof(double));

    m_phase_voc = new PhaseVocoder(m_data_length, m_step_size);

    m_magnitude = new double[m_half_length];
    m_theta_angle = new double[m_half_length];
    m_unwrapped = new double[m_half_length];

    m_window = new Window<double>(HanningWindow, m_data_length);
    m_windowed = new double[m_data_length];
}

void DetectionFunction::deinitialize() {
    delete[] m_mag_history;
    delete[] m_phase_history;
    delete[] m_phase_history_old;
    delete[] m_mag_peaks;

    delete m_phase_voc;

    delete[] m_magnitude;
    delete[] m_theta_angle;
    delete[] m_windowed;
    delete[] m_unwrapped;

    delete m_window;
}

double DetectionFunction::process_time_domain(const std::vector<double>& samples) {
    if (samples.size() != static_cast<size_t>(m_data_length)) {
        // Handle error: input vector size does not match expected frame length
        return 0.0;
    }

    m_window->cut(samples.data(), m_windowed);

    m_phase_voc->processTimeDomain(m_windowed, m_magnitude, m_theta_angle, m_unwrapped);

    if (m_whiten) {
        whiten();
    }

    return run_df();
}

std::vector<double> DetectionFunction::get_spectrum_magnitude() {
    return std::vector<double>(m_magnitude, m_magnitude + m_half_length);
}

void DetectionFunction::whiten() {
    for (int i = 0; i < m_half_length; ++i) {
        double m = m_magnitude[i];
        if (m < m_mag_peaks[i]) {
            m = m + (m_mag_peaks[i] - m) * m_whiten_relax_coeff;
        }
        if (m < m_whiten_floor) m = m_whiten_floor;
        m_mag_peaks[i] = m;
        m_magnitude[i] /= m;
    }
}

double DetectionFunction::run_df() {
    double ret_val = 0.0;

    switch (m_df_type) {
        case DF_HFC:
            ret_val = hfc(m_half_length, m_magnitude);
            break;
        case DF_SPECDIFF:
            ret_val = spec_diff(m_half_length, m_magnitude);
            break;
        case DF_PHASEDEV:
            ret_val = phase_dev(m_half_length, m_theta_angle);
            break;
        case DF_COMPLEXSD:
            ret_val = complex_sd(m_half_length, m_magnitude, m_theta_angle);
            break;
        case DF_BROADBAND:
            ret_val = broadband(m_half_length, m_magnitude);
            break;
    }

    return ret_val;
}

double DetectionFunction::hfc(int length, double* src) {
    double val = 0.0;
    for (int i = 0; i < length; i++) {
        val += src[i] * (i + 1);
    }
    return val;
}

double DetectionFunction::spec_diff(int length, double* src) {
    double val = 0.0;
    double temp = 0.0;
    double diff = 0.0;

    for (int i = 0; i < length; i++) {
        temp = std::fabs((src[i] * src[i]) - (m_mag_history[i] * m_mag_history[i]));
        diff = std::sqrt(temp);
        val += diff;
        m_mag_history[i] = src[i];
    }

    return val;
}

double DetectionFunction::phase_dev(int length, double* src_phase) {
    double tmp_phase = 0.0;
    double tmp_val = 0.0;
    double val = 0.0;
    double dev = 0.0;

    for (int i = 0; i < length; i++) {
        tmp_phase = (src_phase[i] - 2 * m_phase_history[i] + m_phase_history_old[i]);
        dev = MathUtilities::princarg(tmp_phase);
        tmp_val = std::fabs(dev);
        val += tmp_val;
        m_phase_history_old[i] = m_phase_history[i];
        m_phase_history[i] = src_phase[i];
    }

    return val;
}

double DetectionFunction::complex_sd(int length, double* src_magnitude, double* src_phase) {
    double val = 0.0;
    double tmp_phase = 0.0;
    double tmp_real = 0.0;
    double tmp_imag = 0.0;
    double dev = 0.0;
    ComplexData meas(0.0, 0.0);
    ComplexData j(0.0, 1.0);

    for (int i = 0; i < length; i++) {
        tmp_phase = (src_phase[i] - 2 * m_phase_history[i] + m_phase_history_old[i]);
        dev = MathUtilities::princarg(tmp_phase);
        meas = m_mag_history[i] - (src_magnitude[i] * exp(j * dev));
        tmp_real = std::real(meas);
        tmp_imag = std::imag(meas);
        val += std::sqrt((tmp_real * tmp_real) + (tmp_imag * tmp_imag));
        m_phase_history_old[i] = m_phase_history[i];
        m_phase_history[i] = src_phase[i];
        m_mag_history[i] = src_magnitude[i];
    }

    return val;
}

double DetectionFunction::broadband(int length, double* src) {
    double val = 0.0;
    for (int i = 0; i < length; ++i) {
        double sqr_mag = src[i] * src[i];
        if (m_mag_history[i] > 0.0) {
            double diff = 10.0 * std::log10(sqr_mag / m_mag_history[i]);
            if (diff > m_db_rise) val = val + 1.0;
        }
        m_mag_history[i] = sqr_mag;
    }
    return val;
}

