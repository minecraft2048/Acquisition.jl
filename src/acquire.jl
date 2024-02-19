"""
$(SIGNATURES)
Perform the aquisition of multiple satellites with `prns` in system `system` with signal `signal`
sampled at rate `sampling_freq`. Optional arguments are the intermediate frequency `interm_freq`
(default 0Hz), the maximum expected Doppler `max_doppler` (default 7000Hz). If the maximum Doppler
is too unspecific you can instead pass a Doppler range with with your individual step size using
the argument `dopplers`.
"""
function acquire(
    system::AbstractGNSS,
    signal,
    sampling_freq,
    prns::AbstractVector{<:Integer};
    interm_freq = 0.0Hz,
    max_doppler = 7000Hz,
    dopplers = -max_doppler:1/3/(0.001s):max_doppler, #TODO unhardcode this
    noncoherent_rounds = 1,
    compensate_doppler_code = true,
    coherent_integration_length=get_code_length(system)/get_code_frequency(system),
    plan=AcquisitionPlanCPU,
    doppler_offsets = (0.0Hz for _ in prns),
    flip_correlation = false
)

    coherent_integration_length_samples = Int(ceil(upreferred(coherent_integration_length * sampling_freq)))
    acq_plan = plan(
        system,
        coherent_integration_length_samples,
        sampling_freq;
        dopplers,
        prns, 
        fft_flag = FFTW.ESTIMATE,
        compensate_doppler_code = compensate_doppler_code,
        noncoherent_rounds= noncoherent_rounds,
        flip_correlation = flip_correlation
    )
    acquire!(acq_plan, signal, prns; interm_freq, doppler_offsets)
end

"""
$(SIGNATURES)
Perform the aquisition of a single satellite `prn` in system `system` with signal `signal`
sampled at rate `sampling_freq`. Optional arguments are the intermediate frequency `interm_freq`
(default 0Hz), the maximum expected Doppler `max_doppler` (default 7000Hz). If the maximum Doppler
is too unspecific you can instead pass a Doppler range with with your individual step size using
the argument `dopplers`.
"""
function acquire(
    system::AbstractGNSS,
    signal,
    sampling_freq,
    prn::Integer;
    interm_freq = 0.0Hz,
    max_doppler = 7000Hz,
    noncoherent_rounds=1,
    coherent_integration_length=get_code_length(system)/get_code_frequency(system),
    dopplers = -max_doppler:1/3/(coherent_integration_length):max_doppler,
    compensate_doppler_code=true,
    plan=AcquisitionPlanCPU
    )

    only(acquire(system, signal, sampling_freq, [prn]; interm_freq, dopplers, coherent_integration_length,noncoherent_rounds, plan=plan,compensate_doppler_code=compensate_doppler_code))

end

function acquire!(
    acq_plan::AbstractAcquisitionPlan,
    signal,
    prn::Integer;
    interm_freq = 0.0Hz,
    doppler_offset = 0.0Hz,
    noise_power = nothing,
)
    only(acquire!(acq_plan, signal, [prn]; interm_freq, doppler_offset, noise_power))
end


"""
$(SIGNATURES)
Performs a coarse aquisition and fine acquisition of multiple satellites `prns` in system `system` with signal `signal`
sampled at rate `sampling_freq`. The aquisition is performed as parallel code phase
search using the Doppler frequencies with coarse step size `coarse_step` and fine step size `fine_step`.
"""
function coarse_fine_acquire(
    system::AbstractGNSS,
    signal,
    sampling_freq,
    prns::AbstractVector{<:Integer};
    interm_freq = 0.0Hz,
    max_doppler = 7000Hz,
    plan = AcquisitionPlanCPU,
    coherent_integration_length=get_code_length(system)/get_code_frequency(system),
    coarse_step = 1 / 3 / coherent_integration_length,
    fine_step = 1 / 12 / coherent_integration_length,
    noncoherent_rounds=1
)
    coherent_integration_length_samples = Int(ceil(upreferred(coherent_integration_length * sampling_freq)))
    acq_plan = CoarseFineAcquisitionPlan(
        system,
        coherent_integration_length_samples,
        sampling_freq;
        max_doppler,
        coarse_step,
        fine_step,
        prns,
        inner_plan = plan,
        noncoherent_rounds=noncoherent_rounds
    )
    acquire!(acq_plan, signal, prns; interm_freq)
end

"""
$(SIGNATURES)
Performs a coarse aquisition and fine acquisition of a single satellite with PRN `prn` in system `system` with signal `signal`
sampled at rate `sampling_freq`. The aquisition is performed as parallel code phase
search using the Doppler frequencies with coarse step size `coarse_step` and fine step size `fine_step`.
"""
function coarse_fine_acquire(
    system::AbstractGNSS,
    signal,
    sampling_freq,
    prn::Integer;
    interm_freq = 0.0Hz,
    max_doppler = 7000Hz,
    coherent_integration_length=get_code_length(system)/get_code_frequency(system),
    coarse_step = 1 / 3 / coherent_integration_length,
    fine_step = 1 / 12 / coherent_integration_length,
    plan = AcquisitionPlanCPU,
    noncoherent_rounds=1
)
    only(
        coarse_fine_acquire(
            system,
            signal,
            sampling_freq,
            [prn];
            interm_freq,
            max_doppler,
            coarse_step,
            fine_step,
            coherent_integration_length,
            plan,
            noncoherent_rounds=noncoherent_rounds
        ),
    )
end
