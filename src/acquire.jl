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
    plan=AcquisitionPlanCPU
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
        noncoherent_rounds= noncoherent_rounds
    )
    acquire!(acq_plan, signal, prns; interm_freq)
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


"""
$(SIGNATURES)
This acquisition function uses a predefined acquisition plan to accelerate the computing time.
This will be useful, if you have to calculate the acquisition multiple time in a row.
"""
function acquire!(
    acq_plan::AcquisitionPlan,
    signal,
    prns::AbstractVector{<:Integer};
    interm_freq = 0.0Hz,
    doppler_offset = 0.0Hz,
    noise_power = nothing,    
)
    all(map(prn -> prn in acq_plan.avail_prn_channels, prns)) ||
        throw(ArgumentError("You'll need to plan every PRN"))
    code_period = get_code_length(acq_plan.system) / get_code_frequency(acq_plan.system)


#=     powers_per_sats, complex_sigs_per_sats =
        power_over_doppler_and_codes!(acq_plan, signal, prns, interm_freq, doppler_offset) =#

    #Currently we parallelize over doppler taps 
    #It might be faster if we parallelize over PRNs

    #preallocate noncoherent integration workbuffer
    # we want a 3d array with dimensions of (1ms signal * doppler taps * prns)
    plan = acq_plan
    powers_per_sats = [zeros(Float32,plan.signal_length,length(plan.dopplers)) for _ in eachindex(prns)]
    intermediate_freq = interm_freq
    center_frequency = doppler_offset
    noncoherent_rounds = plan.noncoherent_rounds
    compensate_doppler_code = plan.compensate_doppler_code
    for (noncoherent_power_buffer_idx, prn) in enumerate(prns)
        noncoherent_power_accumulator_buffer = powers_per_sats[noncoherent_power_buffer_idx]
        for (chunk,round_idx) in zip(Iterators.partition(signal[1:(noncoherent_rounds*plan.signal_length)],plan.signal_length), 0:noncoherent_rounds-1)
            acq = acquire_mt!(plan,chunk, [prn]; interm_freq=(intermediate_freq+center_frequency))
              #for each dopplers compensate for code drift
              for (acc,power_doppler,dop) in zip(eachcol(noncoherent_power_accumulator_buffer),eachcol(acq[1]),plan.dopplers)
                  #CM-ABS algo 3 step 6 eqn 7
                  doppler1 = ustrip(dop)
                  if compensate_doppler_code == :negative
                      Ns = sign(ustrip(doppler1) - ustrip(center_frequency)) * round(round_idx*0.001* ustrip(plan.sampling_freq) * abs(ustrip(doppler1) - ustrip(center_frequency))/ustrip(get_center_frequency(plan.system)), RoundNearest)
                      acc .= acc .+ circshift(power_doppler,-Ns)
                  elseif compensate_doppler_code == :positive
                      Ns = sign(ustrip(doppler1) - ustrip(center_frequency)) * round(round_idx*0.001* ustrip(plan.sampling_freq) * abs(ustrip(doppler1) - ustrip(center_frequency))/ustrip(get_center_frequency(plan.system)), RoundNearest)
                      acc .= acc .+ circshift(power_doppler,Ns)
                  else
                      acc .= acc .+ power_doppler
                  end
              end
        end
    end

    map(powers_per_sats, prns) do powers, prn
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

function acquire_mt!(
    acq_plan::AcquisitionPlan,
    signal,
    prns::AbstractVector{<:Integer};
    interm_freq = 0.0Hz,
    doppler_offset = 0.0Hz,
    noise_power = nothing,
)
    all(map(prn -> prn in acq_plan.avail_prn_channels, prns)) ||
        throw(ArgumentError("You'll need to plan every PRN"))
    code_period = get_code_length(acq_plan.system) / get_code_frequency(acq_plan.system)

    signal_baseband_buffer, signal_baseband_freq_domain_buffer = preallocate_thread_local_buffer(acq_plan.signal_length, length(acq_plan.dopplers))
    #println(prns)

    powers_per_sats = power_over_dopplers_code_mt!(acq_plan,signal,prns,interm_freq,signal_baseband_buffer,signal_baseband_freq_domain_buffer)
    return powers_per_sats
end

function acquire!(
    acq_plan::CoarseFineAcquisitionPlan,
    signal,
    prns::AbstractVector{<:Integer};
    interm_freq = 0.0Hz,
)
    acq_res = acquire!(acq_plan.coarse_plan, signal, prns; interm_freq)
    map(acq_res, prns) do res, prn
        acquire!(
            acq_plan.fine_plan,
            signal,
            prn;
            interm_freq,
            doppler_offset = res.carrier_doppler,
            noise_power = res.noise_power,
        )
    end
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
    coarse_step = 1 / 3 / (length(signal) / sampling_freq),
    fine_step = 1 / 12 / (length(signal) / sampling_freq),
)
    acq_plan = CoarseFineAcquisitionPlan(
        system,
        length(signal),
        sampling_freq;
        max_doppler,
        coarse_step,
        fine_step,
        prns,
        fft_flag = FFTW.ESTIMATE,
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
    coarse_step = 1 / 3 / (length(signal) / sampling_freq),
    fine_step = 1 / 12 / (length(signal) / sampling_freq),
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
    doppler_offset = 0.0Hz,
    noise_power = nothing,    
)

#= 
This is a rewrite of upstream acquire! 

Previously they are scattered in multiple source files, which makes it hard to follow.
For this implementation we merge the actual correlation code and the acquire! function

TODO don't use this until it passes regression test

=#


    @tracepoint "Sanity checking" all(map(prn -> prn in acq_plan.avail_prn_channels, prns)) ||
        throw(ArgumentError("You'll need to plan every PRN"))
    @tracepoint "Code period" code_period = get_code_length(acq_plan.system) / get_code_frequency(acq_plan.system)

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
    @tracepoint "Setup" begin 
        plan = acq_plan
        intermediate_freq = interm_freq
        center_frequency = doppler_offset
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


    #loop order 1

#=     for (chunk,round_idx) in zip(Iterators.partition(signal[1:(noncoherent_rounds*plan.signal_length)],plan.signal_length), 0:noncoherent_rounds-1)
        for (signal_power_each_prn,code_fd) in zip(signal_powers,plan.codes_freq_domain) #this implies the PRNs


            for (doppler, signal_power_each_doppler) in zip(plan.dopplers, eachcol(signal_power_each_prn))
                downconvert!(plan.signal_baseband, chunk, interm_freq + doppler, plan.sampling_freq) #43*2 downconverts
                mul!(plan.signal_baseband_freq_domain, plan.fft_plan, plan.signal_baseband) # signal_baseband_freq_domain = fft(signal_baseband)
                                                                                            #43*2 ffts
                
                
                @. plan.code_freq_baseband_freq_domain = code_fd * conj(plan.signal_baseband_freq_domain) #This is currently scalar code
                ldiv!(plan.code_baseband, plan.fft_plan, plan.code_freq_baseband_freq_domain) # code_baseband =  ifft(code_freq_baseband_freq_domain)

                #Do doppler compensation using CM-ABS algo 3 step 6 eqn 7

                Ns = sign(ustrip(doppler) - ustrip(center_frequency)) * round(round_idx*0.001* ustrip(plan.sampling_freq) * abs(ustrip(doppler) - ustrip(center_frequency))/ustrip(get_center_frequency(plan.system)), RoundNearest)

                #plan.code_baseband .= abs2.(plan.code_baseband) #will this vectorize???

                
                #signal_power_each_doppler .= signal_power_each_doppler .+ abs2.((circshift(plan.code_baseband, -Ns)))

                #Handwritten loop to do circshift and add simultaneously
                #circshift is a slow operation, it shows up on the profiler

                for i in eachindex(signal_power_each_doppler)
                    signal_power_each_doppler[i] = signal_power_each_doppler[i] + abs2(plan.code_baseband[i])
                end
            end
        end
    end =#


    #loop order 2

    @tracepoint "Correlation" for (chunk,round_idx) in zip(Iterators.partition(signal[1:(noncoherent_rounds*plan.signal_length)],plan.signal_length), 0:noncoherent_rounds-1)
        @tracepoint "Signal chunk" begin
            for (doppler_idx,doppler) in enumerate(plan.dopplers)
                @tracepoint "Doppler taps" begin
                    @tracepoint "Downconvert" downconvert!(plan.signal_baseband, chunk, interm_freq + doppler, plan.sampling_freq) 
                    @tracepoint "Forward FFT" mul!(plan.signal_baseband_freq_domain, plan.fft_plan, plan.signal_baseband)
                    for (signal_power_each_prn,code_fd) in zip(signal_powers,plan.codes_freq_domain)
                        @tracepoint "Inner loop" begin
                            @tracepoint "Complex conjugate"  @. plan.code_freq_baseband_freq_domain = code_fd * conj(plan.signal_baseband_freq_domain) 
                            @tracepoint "Inverse FFT" ldiv!(plan.code_baseband, plan.fft_plan, plan.code_freq_baseband_freq_domain)
                            if round_idx == 1 #For first noncoherent integration we overwrite the output buffer
                                    signal_power_each_prn[:,doppler_idx] .= abs2.(plan.code_baseband)
                            else
                                if plan.compensate_doppler_code != false
                                    Ns = sign(ustrip(doppler) - ustrip(doppler_offset)) * round(round_idx*0.001* ustrip(plan.sampling_freq) * abs(ustrip(doppler) - ustrip(doppler_offset))/ustrip(get_center_frequency(plan.system)), RoundNearest)
                                    signal_power_each_prn[:,doppler_idx] .= signal_power_each_prn[:,doppler_idx] .+ abs2.(circshift(plan.code_baseband,-Ns))
                                else
                                    signal_power_each_prn[:,doppler_idx] .= signal_power_each_prn[:,doppler_idx] .+ abs2.(plan.code_baseband)
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
        

#=         for (signal_power_each_prn,code_fd) in zip(signal_powers,plan.codes_freq_domain) #this implies the PRNs


            for (doppler, signal_power_each_doppler) in zip(plan.dopplers, eachcol(signal_power_each_prn))
                downconvert!(plan.signal_baseband, chunk, interm_freq + doppler, plan.sampling_freq) #43*2 downconverts
                mul!(plan.signal_baseband_freq_domain, plan.fft_plan, plan.signal_baseband) # signal_baseband_freq_domain = fft(signal_baseband)
                                                                                            #43*2 ffts
                
                
                @. plan.code_freq_baseband_freq_domain = code_fd * conj(plan.signal_baseband_freq_domain) #This is currently scalar code
                ldiv!(plan.code_baseband, plan.fft_plan, plan.code_freq_baseband_freq_domain) # code_baseband =  ifft(code_freq_baseband_freq_domain)

                #Do doppler compensation using CM-ABS algo 3 step 6 eqn 7

                Ns = sign(ustrip(doppler) - ustrip(center_frequency)) * round(round_idx*0.001* ustrip(plan.sampling_freq) * abs(ustrip(doppler) - ustrip(center_frequency))/ustrip(get_center_frequency(plan.system)), RoundNearest)

                #plan.code_baseband .= abs2.(plan.code_baseband) #will this vectorize???

                
                #signal_power_each_doppler .= signal_power_each_doppler .+ abs2.((circshift(plan.code_baseband, -Ns)))

                #Handwritten loop to do circshift and add simultaneously
                #circshift is a slow operation, it shows up on the profiler

                for i in eachindex(signal_power_each_doppler)
                    signal_power_each_doppler[i] = signal_power_each_doppler[i] + abs2(plan.code_baseband[i])
                end
            end
        end =#
    end

#=     for (noncoherent_power_buffer_idx, prn) in enumerate(prns)
        noncoherent_power_accumulator_buffer = powers_per_sats[noncoherent_power_buffer_idx]
        for (chunk,round_idx) in zip(Iterators.partition(signal[1:(noncoherent_rounds*plan.signal_length)],plan.signal_length), 0:noncoherent_rounds-1)
            acq = acquire_mt!(plan,chunk, [prn]; interm_freq=(intermediate_freq+center_frequency))
              #for each dopplers compensate for code drift
              for (acc,power_doppler,dop) in zip(eachcol(noncoherent_power_accumulator_buffer),eachcol(acq[1]),plan.dopplers)
                  #CM-ABS algo 3 step 6 eqn 7
                  doppler1 = ustrip(dop)
                  if compensate_doppler_code == :negative
                      Ns = sign(ustrip(doppler1) - ustrip(center_frequency)) * round(round_idx*0.001* ustrip(plan.sampling_freq) * abs(ustrip(doppler1) - ustrip(center_frequency))/ustrip(get_center_frequency(plan.system)), RoundNearest)
                      acc .= acc .+ circshift(power_doppler,-Ns)
                  elseif compensate_doppler_code == :positive
                      Ns = sign(ustrip(doppler1) - ustrip(center_frequency)) * round(round_idx*0.001* ustrip(plan.sampling_freq) * abs(ustrip(doppler1) - ustrip(center_frequency))/ustrip(get_center_frequency(plan.system)), RoundNearest)
                      acc .= acc .+ circshift(power_doppler,Ns)
                  else
                      acc .= acc .+ power_doppler
                  end
              end
        end
    end
 =#


    #Analyse the resulting power bins
    # For GNSS reflectometry application we can skip this, GNSSReflectometry.jl have their own postprocessing algos
    @tracepoint "Postprocessing" map(signal_powers, prns) do powers, prn
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
    doppler_offset = 0.0Hz,
    noise_power = nothing,    
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
        center_frequency = doppler_offset
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

        @tracepoint "Signal chunk" begin

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
                            @tracepoint "Inner loop" begin
                                @no_escape signal_baseband_arena begin
                                signal_baseband = @alloc(ComplexF32, plan.signal_length)
                                downconvert!(signal_baseband, chunk, interm_freq + doppler, plan.sampling_freq)
                                @no_escape signal_baseband_freq_domain_arena begin
                                        signal_baseband_freq_domain = @alloc(ComplexF32,plan.signal_length)
                                        mul!(signal_baseband_freq_domain, plan.forward_fft_plan, signal_baseband)
                                        for (signal_power_each_prn,code_fd) in zip(signal_powers,plan.codes_freq_domain)
                                            @no_escape code_freq_baseband_freq_domain_arena begin
                                                code_freq_baseband_freq_domain = @alloc(ComplexF32, plan.signal_length)
                                                @. code_freq_baseband_freq_domain = code_fd * conj(signal_baseband_freq_domain) 
                                                @no_escape code_baseband_arena begin
                                                    code_baseband = @alloc(ComplexF32,plan.signal_length)
                                                    mul!(code_baseband, plan.inverse_fft_plan, code_freq_baseband_freq_domain)
                                                    #TODO add another arena here #

                                                    if round_idx == 1 #For first noncoherent integration we overwrite the output buffer
                                                        signal_power_each_prn[:,doppler_idx] .= abs2.(code_baseband)
                                                    else
                                                        if plan.compensate_doppler_code != false
                                                            Ns = sign(ustrip(doppler) - ustrip(doppler_offset)) * round(round_idx*0.001* ustrip(plan.sampling_freq) * abs(ustrip(doppler) - ustrip(doppler_offset))/ustrip(get_center_frequency(plan.system)), RoundNearest)
                                                            signal_power_each_prn[:,doppler_idx] .= signal_power_each_prn[:,doppler_idx] .+ abs2.(circshift(code_baseband,-Ns))
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
   map(signal_powers, prns) do powers, prn
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
