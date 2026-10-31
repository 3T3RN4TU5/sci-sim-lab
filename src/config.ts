export const REPO_URL = 'https://github.com/3T3RN4TU5/sci-sim-lab'
export const LIVE_URL = 'https://sci-sim-lab-jeevan.vercel.app'

export interface RouteInfo {
  id: string
  label: string
  short: string
  blurb: string
}

export const ROUTES: RouteInfo[] = [
  { id: 'integrators', label: 'Integrator lab', short: 'ODEs', blurb: 'Euler, RK2, RK4, Dormand–Prince RK45, Verlet, Yoshida-4 and implicit Euler on the pendulum, Kepler and harmonic oscillator. Energy drift, measured convergence order and a stiffness demo.' },
  { id: 'chaos', label: 'Chaos', short: 'Chaos', blurb: 'Lorenz attractor in 3D with a live Benettin Lyapunov exponent, double pendulum trails, sensitivity to initial conditions and the logistic-map bifurcation diagram.' },
  { id: 'nbody', label: 'N-body gravity', short: 'N-body', blurb: 'Barnes–Hut quadtree with quadrupole moments vs direct O(N²) summation: galaxy collision, figure-8 choreography, solar system. Live conservation read-outs and a scaling benchmark.' },
  { id: 'pde', label: 'PDEs', short: 'PDEs', blurb: 'Heat equation (FTCS with CFL check vs Peaceman–Rachford ADI), wave equation with draggable sources and absorbing boundaries, Gray–Scott reaction–diffusion.' },
  { id: 'statphys', label: 'Monte Carlo', short: 'Monte Carlo', blurb: '2D Ising model (Metropolis and Wolff) with a temperature sweep against Onsager’s exact solution, Monte Carlo vs quasi-Monte Carlo error, random-walk diffusion.' },
  { id: 'signals', label: 'Signals', short: 'FFT', blurb: 'Radix-2 FFT from scratch: signal composer, windows, spectrogram of a chirp, windowed-sinc FIR filter design and an optional live microphone spectrum.' },
  { id: 'epidemic', label: 'Epidemics', short: 'SIR', blurb: 'SIR and SEIR models, R₀, final size and herd-immunity threshold, and Nelder–Mead least-squares fitting to noisy synthetic data, animated.' },
]
