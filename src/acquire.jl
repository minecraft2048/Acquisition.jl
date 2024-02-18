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
    acq_plan::AcquisitionPlan,
    signal,
    prn::Integer;
    interm_freq = 0.0Hz,
    doppler_offset = 0.0Hz,
    noise_power = nothing,
)
    only(acquire!(acq_plan, signal, [prn]; interm_freq, doppler_offset, noise_power))
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


function acquire!(acq_plan::CoarseFineAcquisitionPlan, signal, prn::Integer; interm_freq = 0.0Hz)
    only(acquire!(acq_plan, signal, [prn]; interm_freq))
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


"""
$(SIGNATURES)
This acquisition function uses a predefined acquisition plan to accelerate the computing time.
This will be useful, if you have to calculate the acquisition multiple time in a row.
"""
function acquire!(
    acq_plan::AcquisitionPlanCPU,
    signal,
    prns::AbstractVector{<:Integer};
    interm_freq = 0.0Hz,
    doppler_offsets = (0.0Hz for _ in prns),
    noise_powers = nothing,    
)

#= 
This is a rewrite of upstream acquire! 

Previously they are scattered in multiple source files, which makes it hard to follow.
For this implementation we merge the actual correlation code and the acquire! function

TODO don't use this until it passes regression test

=#


     all(map(prn -> prn in acq_plan.avail_prn_channels, prns)) ||
        throw(ArgumentError("You'll need to plan every PRN"))
     code_period = get_code_length(acq_plan.system) / get_code_frequency(acq_plan.system)

    #Currently we parallelize over doppler taps 
    #It might be faster if we parallelize over PRNs, 
    #but currently the noncoherent integration loop is probably not thread safe

    #preallocate noncoherent integration workbuffer
    # we want a 3d array with dimensions of (1ms signal * doppler taps * prns)

    #= 
    read only shared variables:
    signal
    interm_freq
    sampling_freq
    fft_plan
    code_freq_domain

    read only thread local:
    doppler

    read/write buffers:
    signal_baseband
    =#
     begin 
        plan = acq_plan
        intermediate_freq = interm_freq
        noncoherent_rounds = plan.noncoherent_rounds
        compensate_doppler_code = plan.compensate_doppler_code
        signal_powers = plan.signal_powers
    end

    #= Loop count calculation 

        For:
        noncoherent rounds = 1000
        prn = 2
        doppler taps = 43

        Loop order 1:

        for each 1ms signal_chunk (1000)
            for each prns (2)
                for each doppler frequency (43)
                    signal_chunk = downconvert(signal_chunk, doppler) (1000*43*2 = 86000)
                    signal_freq_domain = fft(signal_chunk) (1000*43*2 = 86000)
                    correlation_freq_domain = signal_freq_domain * conj(code_freq_domain) (1000*43*2 = 86000)
                    correlation = ifft(correlation_freq_domain) (1000*43*2 = 86000)
                    power_array = power_array + doppler_compensation() (1000*43*2 = 86000)

        total ops = 5*86000 = 430000


        Loop order 2:
        
        for each 1ms signal_chunk (1000)
            for each doppler frequency (43)
                signal_chunk = downconvert(signal_chunk, doppler) (1000*43 = 43000)
                signal_freq_domain = fft(signal_chunk) (1000*43 = 43000)
                for each prns (2)
                    correlation_freq_domain = signal_freq_domain * conj(code_freq_domain) (1000*43*2 = 86000)
                    correlation = ifft(correlation_freq_domain) (1000*43*2 = 86000)
                    power_array = power_array + doppler_compensation() (1000*43*2 = 86000)

        total ops = 43000+43000+86000+86000+86000 = 344000 but this might have worse memory store locality

        Variables that depends on number of PRNs are:

        signal_powers
        codes_freq_domain

        Variables that depends on number of threads are:
        (currently none)
        
    =#
    #loop order 2

     for (chunk,round_idx) in zip(Iterators.partition(signal[1:(noncoherent_rounds*plan.signal_length)],plan.signal_length), 0:noncoherent_rounds-1)
         begin
            for (doppler_idx,doppler) in enumerate(plan.dopplers)
                 begin
                    for (signal_power_each_prn,code_fd, doppler_offset) in zip(signal_powers,plan.codes_freq_domain, doppler_offsets)
                         begin
                             downconvert!(plan.signal_baseband, chunk, interm_freq + doppler + doppler_offset , plan.sampling_freq) 
                             mul!(plan.signal_baseband_freq_domain, plan.fft_plan, plan.signal_baseband)        
                              @. plan.code_freq_baseband_freq_domain = code_fd * conj(plan.signal_baseband_freq_domain) 
                             ldiv!(plan.code_baseband, plan.fft_plan, plan.code_freq_baseband_freq_domain)
                             begin 
                                if round_idx == 0 #For first noncoherent integration we overwrite the output buffer
                                     signal_power_each_prn[:,doppler_idx] .= abs2.(plan.code_baseband)
                                else
                                    if plan.compensate_doppler_code != false
                                         Ns = sign(ustrip(doppler) - ustrip(doppler_offset)) * round(round_idx*0.001* ustrip(plan.sampling_freq) * abs(ustrip(doppler) - ustrip(doppler_offset))/ustrip(get_center_frequency(plan.system)), RoundNearest)
                                         tmp = circshift(plan.code_baseband,-Ns)
                                         tmp .= abs2.(tmp)
                                         signal_power_each_prn[:,doppler_idx] .= signal_power_each_prn[:,doppler_idx] .+ tmp
                                    else
                                        signal_power_each_prn[:,doppler_idx] .= signal_power_each_prn[:,doppler_idx] .+ abs2.(plan.code_baseband)
                                    end
                                end
                            end
                                #=                             
                                for i in eachindex(plan.code_baseband)

                                @inbounds signal_power_each_prn[i,doppler_idx] = signal_power_each_prn[i,doppler_idx] + abs2(plan.code_baseband[i])
                            end
 =#                     end
                end
            end
        end
    end
        
    end

    #Analyse the resulting power bins
    # For GNSS reflectometry application we can skip this, GNSSReflectometry.jl have their own postprocessing algos

    if isnothing(noise_powers)
        noise_powers = [nothing for _ in prns]
    end

     map(signal_powers, prns, doppler_offsets, noise_powers) do powers, prn, doppler_offset, noise_power
        signal_power, noise_power, code_index, doppler_index = est_signal_noise_power(
            powers,
            acq_plan.sampling_freq,
            get_code_frequency(acq_plan.system), 
            noise_power,
        )
        CN0 = 10 * log10(signal_power / noise_power / code_period / 1.0Hz)
        doppler =
            (doppler_index - 1) * step(acq_plan.dopplers) +
            first(acq_plan.dopplers) +
            doppler_offset
        code_phase =
            (code_index - 1) /
            (acq_plan.sampling_freq / get_code_frequency(acq_plan.system))
        AcquisitionResults(
            acq_plan.system,
            prn,
            acq_plan.sampling_freq,
            doppler,
            code_phase,
            CN0,
            noise_power,
            powers,
            (acq_plan.dopplers .+ doppler_offset) / 1.0Hz,
        )
    end 
end

#= """
$(SIGNATURES)
This acquisition function uses a predefined acquisition plan to accelerate the computing time.
This will be useful, if you have to calculate the acquisition multiple time in a row.
""" =#
function acquire!(
    acq_plan::AcquisitionPlanCPUMultithreaded,
    signal,
    prns::AbstractVector{<:Integer};
    interm_freq = 0.0Hz,
    doppler_offsets = (0.0Hz for _ in prns),
    noise_powers = nothing,    
)

#= 
This is a rewrite of upstream acquire! 

Previously they are scattered in multiple source files, which makes it hard to follow.
For this implementation we merge the actual correlation code and the acquire! function

TODO don't use this until it passes regression test

=#


#=     all(map(prn -> prn in acq_plan.avail_prn_channels, prns)) ||
        throw(ArgumentError("You'll need to plan every PRN"))
 =#    code_period = get_code_length(acq_plan.system) / get_code_frequency(acq_plan.system)

    #Currently we parallelize over doppler taps 
    #It might be faster if we parallelize over PRNs, 
    #but currently the noncoherent integration loop is probably not thread safe

    #preallocate noncoherent integration workbuffer
    # we want a 3d array with dimensions of (1ms signal * doppler taps * prns)

    #= 
    read only shared variables:
    signal
    interm_freq
    sampling_freq
    fft_plan
    code_freq_domain

    read only thread local:
    doppler

    read/write buffers:
    signal_baseband
    =#
    begin 
        plan = acq_plan
        intermediate_freq = interm_freq
        noncoherent_rounds = plan.noncoherent_rounds
        compensate_doppler_code = plan.compensate_doppler_code
        signal_powers = plan.signal_powers
    end


    #n_chunk = length(plan.dopplers) ÷ plan.n_threads
    n_chunk = Int(ceil(length(plan.dopplers) / plan.n_threads))
#=     signal_baseband_arenas_chunk = Iterators.partition(plan.signal_baseband_arenas, n_chunk)
    signal_baseband_freq_domain_arenas_chunk = Iterators.partition(plan.signal_baseband_freq_domain_arenas, n_chunk)
    code_freq_baseband_freq_domain_arenas_chunk = Iterators.partition(plan.code_freq_baseband_freq_domain_arenas, n_chunk)
    code_baseband_arenas_chunk = Iterators.partition(cplan.ode_baseband_arenas, n_chun =#
    doppler_idx_chunk = Iterators.partition(1:length(plan.dopplers), n_chunk)
    doppler_chunk = Iterators.partition(plan.dopplers, n_chunk)
    signal_chunk = Iterators.partition(1:(noncoherent_rounds*plan.signal_length),plan.signal_length)

    for (chunk_idxs,round_idx) in zip(signal_chunk, 0:noncoherent_rounds-1)

        chunk = view(signal, chunk_idxs)

         begin

            @sync begin
                
                for (signal_baseband_arena, 
                    signal_baseband_freq_domain_arena, 
                    code_freq_baseband_freq_domain_arena, 
                    code_baseband_arena, doppler_idx_c, doppler_c) in zip(
                        plan.signal_baseband_arenas, 
                        plan.signal_baseband_freq_domain_arenas, 
                        plan.code_freq_baseband_freq_domain_arenas, 
                        plan.code_baseband_arenas, 
                        doppler_idx_chunk,
                        doppler_chunk)

                    Threads.@spawn begin
                        for (doppler_idx, doppler) in zip(doppler_idx_c, doppler_c)
                             begin
                                for (signal_power_each_prn,code_fd, doppler_offset) in zip(signal_powers,plan.codes_freq_domain,doppler_offsets)
                                    @no_escape signal_baseband_arena begin
                                        signal_baseband = @alloc(ComplexF32, plan.signal_length)
                                        downconvert!(signal_baseband, chunk, interm_freq + doppler + doppler_offset, plan.sampling_freq)
                                        @no_escape signal_baseband_freq_domain_arena begin
                                            signal_baseband_freq_domain = @alloc(ComplexF32,plan.signal_length)
                                            mul!(signal_baseband_freq_domain, plan.forward_fft_plan, signal_baseband)    
                                            @no_escape code_freq_baseband_freq_domain_arena begin
                                                code_freq_baseband_freq_domain = @alloc(ComplexF32, plan.signal_length)
                                                if plan.flip_correlation
                                                    @. code_freq_baseband_freq_domain = conj(code_fd) * signal_baseband_freq_domain
                                                else
                                                    @. code_freq_baseband_freq_domain = code_fd * conj(signal_baseband_freq_domain)
                                                end
                                                @no_escape code_baseband_arena begin
                                                    code_baseband = @alloc(ComplexF32,plan.signal_length)
                                                    mul!(code_baseband, plan.inverse_fft_plan, code_freq_baseband_freq_domain)
                                                    #TODO add another arena here #
                                                    if round_idx == 0 #For first noncoherent integration we overwrite the output buffer
                                                        signal_power_each_prn[:,doppler_idx] .= abs2.(code_baseband)
                                                    else
                                                        if plan.compensate_doppler_code != false
                                                            Ns = sign(ustrip(doppler) + ustrip(doppler_offset)) * round(round_idx*0.001* ustrip(plan.sampling_freq) * abs(ustrip(doppler) + ustrip(doppler_offset))/ustrip(get_center_frequency(plan.system)), RoundNearest)
                                                            signal_power_each_prn[:,doppler_idx] .= signal_power_each_prn[:,doppler_idx] .+ abs2.(circshift(code_baseband,-Ns)) #TODO circshift is allocating
                                                        else
                                                            signal_power_each_prn[:,doppler_idx] .= signal_power_each_prn[:,doppler_idx] .+ abs2.(code_baseband)
                                                        end
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end

            end

        end
    end

    #Analyse the resulting power bins
    # For GNSS reflectometry application we can skip this, GNSSReflectometry.jl have their own postprocessing algos
    if isnothing(noise_powers)
        noise_powers = [nothing for _ in prns]
    end

   map(signal_powers, prns, doppler_offsets, noise_powers) do powers, prn, doppler_offset, noise_power
        signal_power, noise_power, code_index, doppler_index = est_signal_noise_power(
            powers,
            acq_plan.sampling_freq,
            get_code_frequency(acq_plan.system), 
            noise_power,
        )
        CN0 = 10 * log10(signal_power / noise_power / code_period / 1.0Hz)
        doppler =
            (doppler_index - 1) * step(acq_plan.dopplers) +
            first(acq_plan.dopplers) +
            doppler_offset
        code_phase =
            (code_index - 1) /
            (acq_plan.sampling_freq / get_code_frequency(acq_plan.system))
        AcquisitionResults(
            acq_plan.system,
            prn,
            acq_plan.sampling_freq,
            doppler,
            code_phase,
            CN0,
            noise_power,
            powers,
            (acq_plan.dopplers .+ doppler_offset) / 1.0Hz,
        )
    end 
end
