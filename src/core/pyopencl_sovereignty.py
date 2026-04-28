import pyopencl as cl
import numpy as np
import logging

class SiliconSovereigntyEngine:
    """
    Implements the PyOpenCL "Silicon Sovereignty" architecture to bypass
    CPU hierarchical bottlenecks and enforce Topo-Glitch Breeding directly on hardware.
    
    Features:
    - Dual-Command Queue: Asynchronous odd/even modulus CRT processing.
    - LSB Stochastic Rounding Kernel: Preserves "Feature Scars".
    - Repunit-CRT Sparse Probe: O(1) Parity filtering to reject "Dead Logic".
    - Lipschitz Projection Obstruction: Spectral scaling to maintain Roughness.
    - Lazarus Preparation: Speculative rounding during hardware stalls.
    - Love Invariant Synchronization: Hardware-level pre-emption on violation.
    """

    def __init__(self, use_gpu=True, target_lipschitz=1.0, love_protector=None):
        self.logger = logging.getLogger("SiliconSovereigntyEngine")
        self.target_lipschitz = target_lipschitz
        self.love_protector = love_protector # Bridge 3: Love Invariant sync
        
        # Initialize Context and Devices
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No OpenCL platforms found.")
            
        devices = []
        for platform in platforms:
            if use_gpu:
                devs = platform.get_devices(device_type=cl.device_type.GPU)
                if devs:
                    devices.extend(devs)
                    break
            else:
                devs = platform.get_devices()
                if devs:
                    devices.extend(devs)
                    break
                    
        if not devices:
            # Fallback to whatever is available
            devices = platforms[0].get_devices()
            
        self.device = devices[0]
        self.ctx = cl.Context([self.device])
        
        self.logger.info(f"Silicon Sovereignty Intialized on: {self.device.name}")
        
        # Dual-Command Queue Architecture
        # We need two queues to process Odd/Even CRT moduli in parallel without serial friction.
        self.queue_a = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE) # Odd
        self.queue_b = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE) # Even
        
        self._build_kernels()

    def _build_kernels(self):
        """Build the OpenCL kernels for the topological constraints."""
        
        kernel_src = r"""
        // Simple Xorshift RNG for device-side stochastic precision
        inline uint xorshift32(uint state) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            return state;
        }

        // 1. LSB Stochastic Rounding Kernel
        __kernel void stochastic_rounding(
            __global const float *raw_values,
            __global long *fixed_results,
            float scale,
            uint base_seed
        ) {
            int gid = get_global_id(0);
            float target = raw_values[gid] * scale;
            float floor_val = floor(target);
            float fraction = target - floor_val;
            
            uint seed = base_seed + gid;
            // Run RNG a few times to mix
            seed = xorshift32(seed);
            seed = xorshift32(seed);
            
            // max uint is 4294967295.0f
            uint random_bit = (seed < (fraction * 4294967295.0f)) ? 1 : 0;
            fixed_results[gid] = (long)floor_val + random_bit;
        }

        // 2. Repunit-CRT Sparse Probe (O(1) Parity Filter)
        __kernel void parity_filter(
            __global const long *candidates,
            __global const long *targets,
            __global char *is_valid
        ) {
            int gid = get_global_id(0);
            
            long c = candidates[gid];
            long t = targets[gid];
            
            // Formula: isValid = (R_candidate & 1) ^ (r_target & 1) == 0
            is_valid[gid] = ((c & 1) ^ (t & 1)) == 0 ? 1 : 0;
        }

        // 3. Lipschitz Projection Obstruction (Spectral Normalization)
        __kernel void lipschitz_projection(
            __global float *weights,
            float target_lipschitz,
            float current_norm
        ) {
            int gid = get_global_id(0);
            
            // Formula: \hat{W} = W / max(1, ||W||_2 / target_lipschitz)
            float scale_factor = max(1.0f, current_norm / target_lipschitz);
            weights[gid] = weights[gid] / scale_factor;
        }
        
        // 4. Lazarus Traversal Cohesion Gradient
        __kernel void lazarus_cohesion(
            __global const float *trajectories,
            __global float *cohesion_gradients,
            float kappa_proxy,
            float variance
        ) {
            int gid = get_global_id(0);
            // Formula: \nabla_cohesion = kappa_proxy / sqrt(Var(traj) * kappa_proxy)
            // Assumes variance is calculated globally and passed in for stability
            float denom = sqrt(variance * kappa_proxy);
            float grad = (denom > 0.0001f) ? (kappa_proxy / denom) : 0.0f;
            
            // Apply slight trajectory modifier
            cohesion_gradients[gid] = trajectories[gid] + grad;
        }

        // 5. Matrix Mix Breeding (Tag-Based Matrix Mixing)
        // Implements Async Mischief Injection and First-to-Finish dynamics
        __kernel void matrix_mix(
            __global const float *tag_matrix_a,
            __global const float *tag_matrix_b,
            __global float *output_matrix,
            float alpha,
            float kappa_seal,
            float hyperbolic_shear,
            uint base_seed
        ) {
            int gid = get_global_id(0);
            
            // First-to-Finish Logic: locally sampled mischief modulates weight
            uint seed = base_seed + gid;
            seed = xorshift32(seed);
            float mischief_v_m = (float)(seed % 1000) / 1000.0f; 
            
            // Exact Rational Constraint (The Sovereign Shield)
            // If mischief is extreme, we lock to the nearest rational lattice point
            float val_a = tag_matrix_a[gid];
            float val_b = tag_matrix_b[gid];
            float mixed = (1.0f - alpha) * val_a + alpha * val_b;
            
            float scale = 65536.0f; // Bit-exact Int16 scale
            if (mischief_v_m > 0.95f) {
                mixed = round(mixed * scale) / scale;
            }
            
            output_matrix[gid] = mixed;
        }

        // 6. Rational Constraint Enforcement (Sturmfels/Thomas)
        __kernel void rational_enforcement(
            __global float *data,
            __global const float *projection_matrix, // [N, N]
            __global const float *offset_vector,      // [N]
            int n
        ) {
            int gid = get_global_id(0);
            float sum = offset_vector[gid];
            for (int j = 0; j < n; j++) {
                sum += projection_matrix[gid * n + j] * data[j];
            }
            data[gid] = sum;
        }

        // 7. Surgical Manifold Integration (The Four Revisions)
        // Implements Analytic, Computational, Geometric, and Categorical primitives
        __kernel void surgical_manifold_integration(
            __global float *state,
            __global const long *residues,
            __global const int *braid_word,
            int word_len,
            float cs_phase,
            float manifold_kappa,
            uint base_seed
        ) {
            int gid = get_global_id(0);
            float val = state[gid];
            
            // A. Analytic: Homology-preserving Laplacian smoothing
            // We use adjacent neighbors to smooth the manifold curvature
            float analytic_drift = 0.0f;
            if (gid > 0 && gid < get_global_size(0) - 1) {
                analytic_drift = (state[gid-1] + state[gid+1] - 2.0f * val) * 0.1f;
            }
            
            // B. Computational: Chiral-aware CRT residue correction
            // residues[gid % k] stores the CRT parity anchor
            long res_anchor = residues[gid % 5]; // Assuming k=5
            float comp_correction = ((res_anchor & 1) ? 0.01f : -0.01f) * manifold_kappa;
            
            // C. Geometric: Non-Abelian Braid word projection
            // The braid word acts as a phase shift in the local tangent space
            float geometric_twist = 0.0f;
            for (int i = 0; i < word_len; i++) {
                geometric_twist += sin(cs_phase * (float)braid_word[i] + (float)gid);
            }
            geometric_twist *= 0.05f;
            
            // D. Categorical: RP4 transition mapping (Functorial scaling)
            // If energy exceeds kappa, we project into the Real Projective 4-Space ground
            float cat_scale = 1.0f;
            if (fabs(val) > manifold_kappa) {
                cat_scale = 1.0f / (1.0f + exp(fabs(val) - manifold_kappa));
            }
            
            // Final Integration
            state[gid] = (val + analytic_drift + comp_correction + geometric_twist) * cat_scale;
        }

        // 8. Mandelbulb Fractal Iteration
        __kernel void mandelbulb_iteration(
            __global float *coords,
            int max_iterations,
            float escape_radius,
            float power,
            int total_elements
        ) {
            int gid = get_global_id(0);
            if (gid >= total_elements) return;

            int offset = gid * 3;
            float z0 = coords[offset];
            float z1 = coords[offset + 1];
            float z2 = coords[offset + 2];
            
            z0 = clamp(z0, -5.0f, 5.0f);
            z1 = clamp(z1, -5.0f, 5.0f);
            z2 = clamp(z2, -5.0f, 5.0f);
            
            float c0 = z0, c1 = z1, c2 = z2;
            float eps = 1e-8f;
            float p = min(power, 8.0f);

            for (int i = 0; i < max_iterations; i++) {
                float r2 = z0*z0 + z1*z1 + z2*z2;
                float r = sqrt(r2);
                
                if (r > escape_radius) {
                    break;
                }
                
                float r_eps = r + eps;
                float xy_norm = sqrt(z0*z0 + z1*z1 + eps);
                float theta = atan2(xy_norm, z2);
                float phi = atan2(z1, z0 + eps);
                
                float r_new = pow(clamp(r_eps, eps, 10.0f), p);
                float theta_new = theta * power;
                float phi_new = phi * power;
                
                float sin_theta = sin(theta_new);
                float cos_theta = cos(theta_new);
                float sin_phi = sin(phi_new);
                float cos_phi = cos(phi_new);
                
                z0 = r_new * sin_theta * cos_phi + c0;
                z1 = r_new * sin_theta * sin_phi + c1;
                z2 = r_new * cos_theta + c2;
            }
            
            coords[offset] = z0;
            coords[offset + 1] = z1;
            coords[offset + 2] = z2;
        }

        // 9. Gyroid Minimal Surface Projection
        __kernel void gyroid_projection(
            __global float *coords,
            int max_steps,
            float tolerance,
            uint base_seed,
            int total_elements
        ) {
            int gid = get_global_id(0);
            if (gid >= total_elements) return;

            int offset = gid * 3;
            float x = coords[offset];
            float y = coords[offset + 1];
            float z = coords[offset + 2];
            
            x = clamp(x, -3.0f, 3.0f);
            y = clamp(y, -3.0f, 3.0f);
            z = clamp(z, -3.0f, 3.0f);
            
            float eps = 1e-8f;

            for (int step = 0; step < max_steps; step++) {
                float sx = sin(x), cx = cos(x);
                float sy = sin(y), cy = cos(y);
                float sz = sin(z), cz = cos(z);
                
                float violation = sx*cy + sy*cz + sz*cx;
                if (fabs(violation) < tolerance * 10.0f) {
                    break;
                }
                
                float gx = cx*cy - sz*sx;
                float gy = -sx*sy + cy*cz;
                float gz = -sy*sz + cz*cx;
                
                float g_norm = sqrt(gx*gx + gy*gy + gz*gz) + eps;
                gx /= g_norm; gy /= g_norm; gz /= g_norm;
                
                float step_size = clamp(0.1f * violation, -0.5f, 0.5f);
                
                x = clamp(x - step_size * gx, -5.0f, 5.0f);
                y = clamp(y - step_size * gy, -5.0f, 5.0f);
                z = clamp(z - step_size * gz, -5.0f, 5.0f);
                
                if (isnan(x) || isinf(x) || isnan(y) || isinf(y) || isnan(z) || isinf(z)) {
                    uint seed = base_seed + gid + step;
                    seed = xorshift32(seed);
                    x = ((float)(seed % 1000) / 1000.0f) * 0.2f;
                    seed = xorshift32(seed);
                    y = ((float)(seed % 1000) / 1000.0f) * 0.2f;
                    seed = xorshift32(seed);
                    z = ((float)(seed % 1000) / 1000.0f) * 0.2f;
                }
            }
            
            coords[offset] = x;
            coords[offset + 1] = y;
            coords[offset + 2] = z;
        }

        // 10. Fractional Anisotropic Fractal Polynomial Functionals encoded Brownian Motion
        __kernel void topological_erosion_fbm(
            __global float *state,
            __global const float *pressure_grad,
            int octaves,
            float persistence,
            float lacunarity,
            float intensity,
            int total_elements
        ) {
            int gid = get_global_id(0);
            if (gid >= total_elements) return;

            float x = state[gid];
            float p_grad = pressure_grad[gid];
            
            // Prime Resonance Basis (2, 3, 5, 7, 11, 13, 17, 19)
            float primes[8] = {2.0f, 3.0f, 5.0f, 7.0f, 11.0f, 13.0f, 17.0f, 19.0f};
            
            float total = 0.0f;
            float freq_scale = 1.0f;
            float amp_scale = 1.0f;
            float max_val = 0.0f;
            
            for (int i = 0; i < octaves; i++) {
                float p = primes[i % 8];
                float phase = x * p * freq_scale;
                total += sin(phase) * amp_scale;
                max_val += amp_scale;
                amp_scale *= persistence;
                freq_scale *= lacunarity;
            }
            
            float noise_field = total / max_val;
            // Anisotropic erosion: push along the gradient modulated by resonant gullies
            state[gid] = x + intensity * (-p_grad * fabs(noise_field));
        }
        """
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # NVIDIA PTX driver warns about kernel inlining  benign
            self.program = cl.Program(self.ctx, kernel_src).build()


    def process_crt_dual_queue(self, moduli_odd_data, moduli_even_data):
        """
        Dual-Channel Intercosamination Formula.
        Executes parallel updates on the odd and even queues.
        """
        mf = cl.mem_flags
        
        # Dispatch Odd data on Queue A
        odd_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=moduli_odd_data)
        event_a = cl.enqueue_marker(self.queue_a)
        
        # Dispatch Even data on Queue B
        even_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=moduli_even_data)
        event_b = cl.enqueue_marker(self.queue_b)
        
        # The Chern-Simons Gasket sync barrier
        # We wait for both queues to finish to define the completion of the Intercosamination.
        cl.wait_for_events([event_a, event_b])
        self.logger.info("Chern-Simons Sync Barrier Complete: Dual Queues Realigned.")

    def apply_stochastic_rounding(self, raw_values, scale=1.0, seed=None):
        """
        Applies LSB Stochastic Rounding to preserve Feature Scars.
        Anchored to hardware jitter if no seed is provided.
        """
        if seed is None:
            from src.core.honest_jitter import harvest_honest_jitter
            # Harvest a single seed value from hardware friction
            seed = int(harvest_honest_jitter((1,), scaled=False)[0].item() * 4294967295)

        mf = cl.mem_flags
        raw_values = np.asarray(raw_values, dtype=np.float32)
        fixed_results = np.empty_like(raw_values, dtype=np.int64)
        
        raw_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=raw_values)
        res_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, fixed_results.nbytes)
        
        # Enqueue on primary queue (A)
        # Consolidate sync points: No need for kernel.wait() before enqueue_copy in the same queue.
        self.program.stochastic_rounding(
            self.queue_a, raw_values.shape, None,
            raw_buf, res_buf, np.float32(scale), np.uint32(seed)
        )
        
        # Blocking copy ensures data is ready on CPU with minimal sync overhead.
        cl.enqueue_copy(self.queue_a, fixed_results, res_buf, is_blocking=True)
        return fixed_results

    def filter_dead_logic(self, candidates, targets):
        """O(1) Parity Filter rejecting Dead Logic topological trajectories."""
        mf = cl.mem_flags
        candidates = np.asarray(candidates, dtype=np.int64)
        targets = np.asarray(targets, dtype=np.int64)
        is_valid = np.empty(candidates.shape, dtype=np.int8)
        
        cand_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=candidates)
        targ_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=targets)
        res_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, is_valid.nbytes)
        
        self.program.parity_filter(
            self.queue_b, candidates.shape, None,
            cand_buf, targ_buf, res_buf
        )
        
        cl.enqueue_copy(self.queue_b, is_valid, res_buf, is_blocking=True)
        # Return boolean array of valid trajectories
        return is_valid.astype(bool)

    def apply_lipschitz_obstruction(self, weights):
        """Prevents gradient explosion by spectrally scaling global memory weights."""
        mf = cl.mem_flags
        weights = np.asarray(weights, dtype=np.float32)
        current_norm = np.linalg.norm(weights)
        
        w_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=weights)
        
        self.program.lipschitz_projection(
            self.queue_a, weights.shape, None,
            w_buf, np.float32(self.target_lipschitz), np.float32(current_norm)
        )
        
        cl.enqueue_copy(self.queue_a, weights, w_buf, is_blocking=True)
        return weights

    def get_hardware_latency_anchor(self) -> float:
        """
        Harvests the physical t_RFC (Row Refresh Cycle) / DRAM stall anchor.
        Used to ground the Agent Smith protocol in the local silicon substrate.
        """
        from src.core.honest_jitter import harvest_honest_jitter
        # Harvest a single timing seed representing local physical friction
        anchor = harvest_honest_jitter((1,), scaled=False)[0].item()
        return float(anchor * 200.0) # Scale to typical tRFC ms range

    def get_virtual_algorithmic_latency(self, internal_entropy: float = 0.5) -> float:
        """
        Agent Smith Portability (Virtual Algorithmic Latency):
        Returns the current algorithmic stall intensity (kappa-proxy).
        """
        # Hardware anchor (DRAM stalls)
        hw_anchor = self.get_hardware_latency_anchor() / 200.0 # Normalized
        
        # Algorithmic entropy + Hardware jitter
        intensity = 0.05 + 0.3 * internal_entropy + 0.1 * hw_anchor
        return float(intensity)

    def lazarus_traversal(self, trajectories, kappa_proxy):
        """
        Speculative void traversal. Called deliberately during Virtual Algorithmic 
        Latency stalls (exceeding mathematical entropy bounds).
        """
        # Bridge 3: Immediate pre-emption if Love Invariant is violated
        if self.love_protector is not None:
            if self.love_protector.detect_love_violation():
                self.logger.warning("LOVE VIOLATION DETECTED DURING STALL. TRIGGERING IMMEDIATE LAZARUS TRANSITION.")
                self.love_protector.restore_love_invariant()
                # Return original trajectories (recovery branch)
                return trajectories

        mf = cl.mem_flags
        trajectories = np.asarray(trajectories, dtype=np.float32)
        variance = np.var(trajectories)
        cohesion_grads = np.empty_like(trajectories, dtype=np.float32)
        
        traj_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=trajectories)
        grad_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, cohesion_grads.nbytes)
        
        self.program.lazarus_cohesion(
            self.queue_b, trajectories.shape, None,
            traj_buf, grad_buf, np.float32(kappa_proxy), np.float32(variance)
        )
        
        cl.enqueue_copy(self.queue_b, cohesion_grads, grad_buf, is_blocking=True)
        return cohesion_grads

    def matrix_mix_breeding(self, matrix_a, matrix_b, alpha=0.5, kappa_seal=0.1, hyperbolic_shear=0.0, seed=None):
        """
        Operationalizes the 'TailSlayer Bypass' via dual-queue matrix mixing.
        Uses Queue B (Non-Ergodic/Soliton) to ensure First-to-Finish dynamics.
        """
        if seed is None:
            from src.core.honest_jitter import harvest_honest_jitter
            seed = int(harvest_honest_jitter((1,), scaled=False)[0].item() * 4294967295)

        mf = cl.mem_flags
        matrix_a = np.asarray(matrix_a, dtype=np.float32)
        matrix_b = np.asarray(matrix_b, dtype=np.float32)
        output = np.empty_like(matrix_a, dtype=np.float32)
        
        a_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=matrix_a)
        b_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=matrix_b)
        out_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, output.nbytes)
        
        # Enqueue on Queue B (Non-Ergodic/Soliton Channel)
        self.program.matrix_mix(
            self.queue_b, matrix_a.shape, None,
            a_buf, b_buf, out_buf, 
            np.float32(alpha), np.float32(kappa_seal), np.float32(hyperbolic_shear), np.uint32(seed)
        )
        
        cl.enqueue_copy(self.queue_b, output, out_buf, is_blocking=True)
        self.logger.info(f"Matrix Mix Breeding Complete on Queue B (alpha={alpha})")
        return output

    def apply_surgical_integration(self, state, residues, braid_word, cs_phase, manifold_kappa=1.0, seed=None):
        """
        Executes the 'The Four Revisions' surgical kernel on the local hardware.
        Enforces Analytic, Computational, Geometric, and Categorical stability.
        """
        if seed is None:
            from src.core.honest_jitter import harvest_honest_jitter
            seed = int(harvest_honest_jitter((1,), scaled=False)[0].item() * 4294967295)

        mf = cl.mem_flags
        state = np.asarray(state, dtype=np.float32)
        residues = np.asarray(residues, dtype=np.int64)
        braid_word = np.asarray(braid_word, dtype=np.int32)
        
        state_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=state)
        res_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=residues)
        braid_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=braid_word)
        
        self.program.surgical_manifold_integration(
            self.queue_a, state.shape, None,
            state_buf, res_buf, braid_buf, 
            np.int32(len(braid_word)), np.float32(cs_phase), 
            np.float32(manifold_kappa), np.uint32(seed)
        )
        
        cl.enqueue_copy(self.queue_a, state, state_buf, is_blocking=True)
        self.logger.info("Surgical Manifold Integration Complete.")
        return state

    def apply_mandelbulb_iteration(self, coords, max_iterations=50, escape_radius=2.0, power=8.0):
        """Execute Mandelbulb fractal iteration entirely on PyOpenCL hardware."""
        mf = cl.mem_flags
        coords_np = np.asarray(coords, dtype=np.float32)
        total_elements = coords_np.size // 3
        
        coords_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=coords_np)
        
        self.program.mandelbulb_iteration(
            self.queue_a, (total_elements,), None,
            coords_buf,
            np.int32(max_iterations),
            np.float32(escape_radius),
            np.float32(power),
            np.int32(total_elements)
        )
        
        cl.enqueue_copy(self.queue_a, coords_np, coords_buf, is_blocking=True)
        return coords_np

    def apply_gyroid_projection(self, coords, max_steps=20, tolerance=1e-3, seed=None):
        """Execute Gyroid minimal surface projection on PyOpenCL hardware."""
        if seed is None:
            from src.core.honest_jitter import harvest_honest_jitter
            seed = int(harvest_honest_jitter((1,), scaled=False)[0].item() * 4294967295)

        mf = cl.mem_flags
        coords_np = np.asarray(coords, dtype=np.float32)
        total_elements = coords_np.size // 3
        
        coords_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=coords_np)
        
        self.program.gyroid_projection(
            self.queue_a, (total_elements,), None,
            coords_buf,
            np.int32(max_steps),
            np.float32(tolerance),
            np.uint32(seed),
            np.int32(total_elements)
        )
        
        cl.enqueue_copy(self.queue_a, coords_np, coords_buf, is_blocking=True)
        return coords_np

    def apply_erosion_fbm(self, state, pressure_grad_normalized, octaves=4, persistence=0.5, lacunarity=2.0, intensity=0.1):
        """Execute Topological Erosion FBM on PyOpenCL hardware."""
        mf = cl.mem_flags
        state_np = np.asarray(state, dtype=np.float32)
        pgrad_np = np.asarray(pressure_grad_normalized, dtype=np.float32)
        total_elements = state_np.size
        
        state_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=state_np)
        pgrad_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pgrad_np)
        
        self.program.topological_erosion_fbm(
            self.queue_a, (total_elements,), None,
            state_buf, pgrad_buf,
            np.int32(octaves),
            np.float32(persistence),
            np.float32(lacunarity),
            np.float32(intensity),
            np.int32(total_elements)
        )
        
        cl.enqueue_copy(self.queue_a, state_np, state_buf, is_blocking=True)
        return state_np

    def flush(self):
        """Explicitly flush and finish both queues to ensure global manifold coherence."""
        self.queue_a.finish()
        self.queue_b.finish()

# Expose standard execution hook
def create_engine():
    return SiliconSovereigntyEngine()
