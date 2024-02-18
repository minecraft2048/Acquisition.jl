abstract type AbstractAcquisitionPlan end

struct AcquisitionPlan{S,DS,CS,P,PS}
    system::S
    signal_length::Int
    sampling_freq::typeof(1.0Hz)
    dopplers::DS
    codes_freq_domain::CS
    signal_baseband::Vector{ComplexF32}
    signal_baseband_freq_domain::Vector{ComplexF32}
    code_freq_baseband_freq_domain::Vector{ComplexF32}
    code_baseband::Vector{ComplexF32}
    signal_powers::Vector{Matrix{Float32}}
    complex_signal::Vector{Matrix{ComplexF32}}
    fft_plan::P
    avail_prn_channels::PS
    compensate_doppler_code::Symbol
    noncoherent_rounds::Int
end

function AcquisitionPlan(
    system,
    signal_length,
    sampling_freq;
    max_doppler = 7000Hz,
    dopplers = -max_doppler:1/3/(signal_length/sampling_freq):max_doppler,
    prns = 1:34,
    fft_flag = FFTW.MEASURE,
    compensate_doppler_code = :disabled,
    noncoherent_rounds = 1
)
    signal_baseband,
    signal_baseband_freq_domain,
    code_freq_baseband_freq_domain,
    code_baseband,
    codes_freq_domain,
    fft_plan = common_buffers(system, signal_length, sampling_freq, prns, fft_flag)
    Δt = signal_length / sampling_freq
    code_interval = get_code_length(system) / get_code_frequency(system)
    signal_powers = [
        Matrix{Float32}(
            undef,
            ceil(Int, sampling_freq * min(Δt, code_interval)),
            length(dopplers),
        ) for _ in prns
    ]
    complex_signal = [
        Matrix{ComplexF32}(
            undef,
            ceil(Int, sampling_freq * min(Δt, code_interval)),
            length(dopplers),
        ) for _ in prns
    ]
    @assert length(signal_powers) == length(complex_signal)
    AcquisitionPlan(
        system,
        signal_length,
        sampling_freq,
        dopplers,
        codes_freq_domain,
        signal_baseband,
        signal_baseband_freq_domain,
        code_freq_baseband_freq_domain,
        code_baseband,
        signal_powers,
        complex_signal,
        fft_plan,
        prns,
        compensate_doppler_code,
        noncoherent_rounds
    )
end

function Base.show(io::IO, ::MIME"text/plain", plan::AcquisitionPlan)
    println(io,"AcquisitionPlan for GPS L1:\n")
    println(io,"Sample rate: $(plan.sampling_freq)")
    println(io,"Doppler taps: $(plan.dopplers)")
    println(io,"Coherent integration length: $(plan.signal_length) samples ($(upreferred(plan.signal_length/plan.sampling_freq)))")
    println(io,"Noncoherent integration round: $(plan.noncoherent_rounds) ($(upreferred(plan.signal_length*plan.noncoherent_rounds/plan.sampling_freq)))")
    println(io, "Doppler compensation: $(plan.compensate_doppler_code)")
    println(io,"FFTW plan: $(plan.fft_plan)")
end

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



#=     signal_baseband,
    signal_baseband_freq_domain,
    code_freq_baseband_freq_domain,
    code_baseband,
    codes_freq_domain,
    fft_plan = common_buffers(system, signal_length, sampling_freq, prns, fft_flag)
    Δt = signal_length / sampling_freq
    code_interval = get_code_length(system) / get_code_frequency(system)
    coarse_signal_powers = [
        Matrix{Float32}(
            undef,
            ceil(Int, sampling_freq * min(Δt, code_interval)),
            length(coarse_dopplers),
        ) for _ in prns
    ]
    
    fine_signal_powers = [
        Matrix{Float32}(
            undef,
            ceil(Int, sampling_freq * min(Δt, code_interval)),
            length(fine_doppler_range),
        ) for _ in prns
    ]
    coarse_signal_powers_complex = [
        Matrix{ComplexF32}(
            undef,
            ceil(Int, sampling_freq * min(Δt, code_interval)),
            length(coarse_dopplers),
        ) for _ in prns
    ]
    fine_signal_powers_complex = [
        Matrix{ComplexF32}(
            undef,
            ceil(Int, sampling_freq * min(Δt, code_interval)),
            length(fine_doppler_range),
        ) for _ in prns
    ]


    coarse_plan = AcquisitionPlan(
        system,
        signal_length,
        sampling_freq,
        coarse_dopplers,
        codes_freq_domain,
        signal_baseband,
        signal_baseband_freq_domain,
        code_freq_baseband_freq_domain,
        code_baseband,
        coarse_signal_powers,
        coarse_signal_powers_complex,
        fft_plan,
        prns,
        :disabled,
        noncoherent_rounds
    )
    fine_plan = AcquisitionPlan(
        system,
        signal_length,
        sampling_freq,
        fine_doppler_range,
        codes_freq_domain,
        signal_baseband,
        signal_baseband_freq_domain,
        code_freq_baseband_freq_domain,
        code_baseband,
        fine_signal_powers,
        fine_signal_powers_complex,
        fft_plan,
        prns,
        :disabled,
        noncoherent_rounds
    ) =#
    CoarseFineAcquisitionPlan(coarse_plan, fine_plan)
end

function common_buffers(system, signal_length, sampling_freq, prns, fft_flag)
    codes = [gen_code(signal_length, system, sat_prn, sampling_freq) for sat_prn in prns]
    signal_baseband = Vector{ComplexF32}(undef, signal_length)
    signal_baseband_freq_domain = similar(signal_baseband)
    code_freq_baseband_freq_domain = similar(signal_baseband)
    code_baseband = similar(signal_baseband)
    fft_plan = plan_fft(signal_baseband; flags = fft_flag)
    codes_freq_domain = map(code -> fft_plan * code, codes)
    signal_baseband,
    signal_baseband_freq_domain,
    code_freq_baseband_freq_domain,
    code_baseband,
    codes_freq_domain,
    fft_plan
end

function preallocate_thread_local_buffer(signal_length,doppler_taps)
    #TODO: align buffers to fftw friendly offsets
    #DO NOT merge this to upstream JuliaGNSS until this is stopped hard coded
    signal_length = 16368
    signal_baseband_buffer = Array{ComplexF32}(undef, signal_length,doppler_taps)
    signal_baseband_freq_domain_buffer = similar(signal_baseband_buffer)

    return signal_baseband_buffer,signal_baseband_freq_domain_buffer

end

# New implementation 

struct AcquisitionPlanCPU{S,DS,CS,P,PS} <: AbstractAcquisitionPlan
    system::S
    signal_length::Int
    sampling_freq::typeof(1.0Hz)
    dopplers::DS
    codes_freq_domain::CS
    signal_baseband::Vector{ComplexF32}
    signal_baseband_freq_domain::Vector{ComplexF32}
    code_freq_baseband_freq_domain::Vector{ComplexF32}
    code_baseband::Vector{ComplexF32}
    signal_powers::Vector{Matrix{Float32}}
    fft_plan::P
    avail_prn_channels::PS
    compensate_doppler_code::Bool
    noncoherent_rounds::Int
end

function AcquisitionPlanCPU(
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
#=     signal_baseband,
    signal_baseband_freq_domain,
    code_freq_baseband_freq_domain,
    code_baseband,
    codes_freq_domain,
    fft_plan = common_buffers(system, signal_length, sampling_freq, prns, fft_flag)

 =#

    #We merge common_buffers here, its a 'misleading' function as it actually do a lot of things at once 
    #instead of just allocating several work buffers

    #Preallocate baseband buffers and resulting correlator output

    signal_baseband = Vector{ComplexF32}(undef, signal_length)
    signal_baseband_freq_domain = similar(signal_baseband)
    code_freq_baseband_freq_domain = similar(signal_baseband)
    signal_baseband_freq_domain = similar(signal_baseband)
    code_baseband = similar(signal_baseband)
    fft_plan = plan_fft(signal_baseband; flags = fft_flag)

    codes = [gen_code(signal_length, system, sat_prn, sampling_freq) for sat_prn in prns]
    codes_freq_domain = [fft_plan * code for code in codes]

    Δt = signal_length / sampling_freq
    code_interval = get_code_length(system) / get_code_frequency(system)
#=     signal_powers = [
        Matrix{Float32}(
            undef,
            ceil(Int, sampling_freq * min(Δt, code_interval)),
            length(dopplers),
        ) for _ in prns
    ]
 =#

    signal_powers = [
            zeros(Float32,
                ceil(Int, sampling_freq * min(Δt, code_interval)),
                length(dopplers),
            ) for _ in prns
        ]






    AcquisitionPlanCPU(
        system,
        signal_length,
        sampling_freq,
        dopplers,
        codes_freq_domain,
        signal_baseband,
        signal_baseband_freq_domain,
        code_freq_baseband_freq_domain,
        code_baseband,
        signal_powers,
        fft_plan,
        prns,
        compensate_doppler_code,
        noncoherent_rounds
    )
end


struct AcquisitionPlanCPUMultithreaded{S,DS,CS,P,P2,PS,T} <: AbstractAcquisitionPlan where T
    system::S
    signal_length::Int
    sampling_freq::typeof(1.0Hz)
    dopplers::DS
    codes_freq_domain::CS
    signal_baseband_arenas::T
    signal_baseband_freq_domain_arenas::T
    code_freq_baseband_freq_domain_arenas::T
    code_baseband_arenas::T
    signal_powers::Vector{Matrix{Float32}}
    forward_fft_plan::P
    inverse_fft_plan::P2
    avail_prn_channels::PS
    compensate_doppler_code::Bool
    noncoherent_rounds::Int
    n_threads::Int
    flip_correlation::Bool
end


function AcquisitionPlanCPUMultithreaded(
    system,
    signal_length,
    sampling_freq;
    max_doppler = 7000Hz,
    dopplers = -max_doppler:1/3/(signal_length/sampling_freq):max_doppler,
    prns = 1:34,
    fft_flag = FFTW.MEASURE,
    compensate_doppler_code = false,
    noncoherent_rounds = 1,
    n_threads=Threads.nthreads(),
    flip_correlation= false,
)
#=     signal_baseband,
    signal_baseband_freq_domain,
    code_freq_baseband_freq_domain,
    code_baseband,
    codes_freq_domain,
    fft_plan = common_buffers(system, signal_length, sampling_freq, prns, fft_flag)

 =#

    #We merge common_buffers here, its a 'misleading' function as it actually do a lot of things at once 
    #instead of just allocating several work buffers

    #Preallocate baseband buffers and resulting correlator output



    signal_baseband_arenas = [
        AllocBuffer() for _ in 1:n_threads
    ]
#=     signal_baseband_freq_domain = similar(signal_baseband)
    code_freq_baseband_freq_domain = similar(signal_baseband)
    signal_baseband_freq_domain = similar(signal_baseband) =#

    signal_baseband_freq_domain_arenas = [
        AllocBuffer() for _ in 1:n_threads
    ]

    code_freq_baseband_freq_domain_arenas = [
        AllocBuffer() for _ in 1:n_threads
    ]

    code_baseband_arenas = [
        AllocBuffer() for _ in 1:n_threads
    ]
    
    forward_fft_plan = plan_fft(Vector{ComplexF32}(undef, signal_length); flags = fft_flag)
    inverse_fft_plan = plan_ifft(Vector{ComplexF32}(undef, signal_length); flags = fft_flag)

    codes = [gen_code(signal_length, system, sat_prn, sampling_freq) for sat_prn in prns]
    codes_freq_domain = [forward_fft_plan * code for code in codes]

    Δt = signal_length / sampling_freq
    code_interval = get_code_length(system) / get_code_frequency(system)
#=     signal_powers = [
        Matrix{Float32}(
            undef,
            ceil(Int, sampling_freq * min(Δt, code_interval)),
            length(dopplers),
        ) for _ in prns
    ]
 =#

    signal_powers = [
            zeros(Float32,
                ceil(Int, sampling_freq * min(Δt, code_interval)),
                length(dopplers),
            ) for _ in prns
        ]


        AcquisitionPlanCPUMultithreaded(
        system,
        signal_length,
        sampling_freq,
        dopplers,
        codes_freq_domain,
        signal_baseband_arenas,
        signal_baseband_freq_domain_arenas,
        code_freq_baseband_freq_domain_arenas,
        code_baseband_arenas,
        signal_powers,
        forward_fft_plan,
        inverse_fft_plan,
        prns,
        compensate_doppler_code,
        noncoherent_rounds,
        n_threads,
        flip_correlation
    )
end



function Base.show(io::IO, ::MIME"text/plain", plan::AcquisitionPlanCPU)
    println(io,"Acquisition plan for GPS L1:\n")
    println(io,"Implementation: Single threaded CPU")
    println(io,"Planned PRNs: $(plan.avail_prn_channels)")
    println(io,"Sample rate: $(plan.sampling_freq)")
    println(io,"Doppler taps: $(plan.dopplers)")
    println(io,"Coherent integration length: $(plan.signal_length) samples ($(upreferred(plan.signal_length/plan.sampling_freq)))")
    println(io,"Noncoherent integration round: $(plan.noncoherent_rounds) ($(upreferred(plan.signal_length*plan.noncoherent_rounds/plan.sampling_freq)))")
    println(io, "Doppler compensation: $(plan.compensate_doppler_code)")
    println(io,"FFTW plan: $(plan.fft_plan)")
end

function Base.show(io::IO, ::MIME"text/plain", plan::AcquisitionPlanCPUMultithreaded)
    println(io,"Acquisition plan for GPS L1:\n")
    println(io,"Implementation: Multithreaded CPU")
    println(io, "Number of threads: $(plan.n_threads)")
    println(io,"Planned PRNs: $(plan.avail_prn_channels)")
    println(io,"Sample rate: $(plan.sampling_freq)")
    println(io,"Doppler taps: $(plan.dopplers)")
    println(io,"Coherent integration length: $(plan.signal_length) samples ($(upreferred(plan.signal_length/plan.sampling_freq)))")
    println(io,"Noncoherent integration round: $(plan.noncoherent_rounds) ($(upreferred(plan.signal_length*plan.noncoherent_rounds/plan.sampling_freq)))")
    println(io, "Doppler compensation: $(plan.compensate_doppler_code)")
    println(io,"FFTW plan: $(plan.forward_fft_plan)")
end

function Base.show(io::IO, ::MIME"text/plain", plan::CoarseFineAcquisitionPlan)
    println(io,"Coarse fine acquisition plan for GPS L1:\n")
    println(io,"Coarse acquisition plan:")
    Base.show(io, "text/plain", plan.coarse_plan)
    println(io,"\nFine acquisition plan:")
    Base.show(io, "text/plain", plan.fine_plan)

end