struct CoarseFineAcquisitionPlan{C<:AbstractAcquisitionPlan,F<:AbstractAcquisitionPlan}
    coarse_plan::C
    fine_plan::F
end

function CoarseFineAcquisitionPlan(
    system,
    signal_length,
    sampling_freq;
    max_doppler = 7000Hz,
    prns = 1:34,
    fft_flag = FFTW.MEASURE,
    noncoherent_rounds=1,
    inner_plan=AcquisitionPlanCPU,
    compensate_doppler_code=true,
    coherent_integration_length = get_code_length(system)/get_code_frequency(system),
    coarse_step = 1 / 3 / coherent_integration_length,
    fine_step = 1 / 12 / coherent_integration_length,

)
    coarse_dopplers = -max_doppler:coarse_step:max_doppler
    fine_doppler_range = -2*coarse_step:fine_step:2*coarse_step

    coarse_plan = inner_plan(
        system,
        signal_length,
        sampling_freq;
        dopplers=coarse_dopplers,
        prns=prns,
        fft_flag=fft_flag,
        noncoherent_rounds=noncoherent_rounds,
        compensate_doppler_code=compensate_doppler_code
    )

    fine_plan = inner_plan(
        system,
        signal_length,
        sampling_freq;
        dopplers=fine_doppler_range,
        prns=prns,
        fft_flag=fft_flag,
        noncoherent_rounds=noncoherent_rounds,
        compensate_doppler_code=compensate_doppler_code
    )

    CoarseFineAcquisitionPlan(coarse_plan, fine_plan)
end

function acquire!(
    acq_plan::CoarseFineAcquisitionPlan,
    signal,
    prns::AbstractVector{<:Integer};
    interm_freq = 0.0Hz,
)
    acq_res = acquire!(acq_plan.coarse_plan, signal, prns; interm_freq)

    doppler_offsets = [res.carrier_doppler for res in acq_res]
    noise_powers = [res.noise_power for res in acq_res]

    acquire!(
        acq_plan.fine_plan,
        signal,
        prns;
        interm_freq,
        doppler_offsets = doppler_offsets,
        noise_powers = noise_powers,
    )

end

function Base.show(io::IO, ::MIME"text/plain", plan::CoarseFineAcquisitionPlan)
    println(io,"Coarse fine acquisition plan for GPS L1:\n")
    println(io,"Coarse acquisition plan:")
    Base.show(io, "text/plain", plan.coarse_plan)
    println(io,"\nFine acquisition plan:")
    Base.show(io, "text/plain", plan.fine_plan)

end