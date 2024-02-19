#=
Template for adding new GNSS signal acquisition algorithms and implementation

Previously upstream Acquisition.jl are structured like this:

acquire.jl: Top level drivers (acquire, acquire!)
calc_powers.jl: Algorithm implementation 
plan_acquire.jl: Generate acquisition plans

This is hard to follow, as a single acquisition algorithm and implementation spans 3 different 
source files and 5+ function calls



This takes inspiration from existing real-time DSP framework structure:

Arduino: https://roboticsbackend.com/arduino-setup-loop-functions-explained/
GNU Radio: https://www.gnuradio.org/doc/doxygen/classgr_1_1block.html
JUCE: https://docs.juce.com/master/tutorial_dsp_introduction.html


DSP framework    | Setup processor block   |     Process signals      | 

Arduino            setup()                  loop()
GNU Radio          gr::block()              gr::general_work()
JUCE               prepare()                process()

=#


#=

AcquisitionPlan struct definition here

struct AcquisitionPlanXYZ <: AbstractAcquisitionPlan
    necessary stuff here
end

=#



#=

Outer constructor here

function AcquisitionPlanXYZ(
    system,
    signal_length,
    sampling_freq;
    max_doppler = 7000Hz,
    dopplers = -max_doppler:1/3/(signal_length/sampling_freq):max_doppler,
    prns = 1:34,
    fft_flag = FFTW.MEASURE,
    compensate_doppler_code = false,
    noncoherent_rounds = 1,
    flip_correlation = false
)

=#

#=

Specific acquisition implementation here

function acquire!(
    acq_plan::AcquisitionPlanXYZ
    signal,
    prns::AbstractVector{<:Integer};
    interm_freq 0.0Hz,
    doppler_offsets = (0.0Hz for _ in prns),
    noise_powers = nothing,    
)


=#