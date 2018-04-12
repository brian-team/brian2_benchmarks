import os
import shutil
import timeit

from brian2.only import *
import brian2.numpy_ as np
from brian2.equations.equations import parse_string_equations
from brian2.parsing.statements import parse_statement
from brian2.parsing.sympytools import sympy_to_str
try:
    from brian2.parsing.sympytools import _str_to_sympy
except ImportError:
    pass


def _clear_cache():
    # Clear the internal cache
    try:
        make_statements._cache.clear()
        parse_string_equations._cache.clear()
        parse_statement._cache.clear()
        _str_to_sympy._cache.clear()
        sympy_to_str._cache.clear()
        StateUpdateMethod.apply_stateupdater._cache.clear()
    except (NameError, AttributeError):
        pass


def codegen_setup(target):
    if target in ['numpy', 'weave', 'cython']:
        set_device('runtime')
        prefs.codegen.target = target
    else:
        set_device(target)
        device.reinit()
        device.activate(directory=None)
    _clear_cache()
    # A bit of a brute-force approach to delete the disk cache
    try:
        if target == 'weave':
            shutil.rmtree(os.path.expanduser(os.path.join('~', '.python27_compiled')))
            shutil.rmtree(os.path.expanduser(os.path.join('~', '.python36_compiled')))
            shutil.rmtree(os.path.expanduser(os.path.join('~', '.cache', 'scipy', 'python27_compiled')))
            shutil.rmtree(os.path.expanduser(os.path.join('~', '.cache', 'scipy', 'python36_compiled')))
        elif target == 'cython':
            shutil.rmtree(os.path.expanduser(os.path.join('~', '.cython', 'brian_extensions')))
    except OSError:
        pass


class FullExamples:
    def setup(self, target):
        codegen_setup(target)

    def time_reliability_example(self, target):
        N = 25
        tau_input = 5 * ms
        input = NeuronGroup(1,
                            'dx/dt = -x / tau_input + (2 /tau_input)**.5 * xi : 1')

        # The noisy neurons receiving the same input
        tau = 10 * ms
        sigma = .015
        eqs_neurons = '''
        dx/dt = (0.9 + .5 * I - x) / tau + sigma * (2 / tau)**.5 * xi : 1
        I : 1 (linked)
        '''
        neurons = NeuronGroup(N, model=eqs_neurons, threshold='x > 1',
                              reset='x = 0', refractory=5 * ms, method='euler')
        neurons.x = 'rand()'
        neurons.I = linked_var(input,
                               'x')  # input.x is continuously fed into neurons.I
        spikes = SpikeMonitor(neurons)

        run(1 * second)

    def time_CUBA_example(self, target):
        taum = 20 * ms
        taue = 5 * ms
        taui = 10 * ms
        Vt = -50 * mV
        Vr = -60 * mV
        El = -49 * mV

        eqs = '''
        dv/dt  = (ge+gi-(v-El))/taum : volt (unless refractory)
        dge/dt = -ge/taue : volt
        dgi/dt = -gi/taui : volt
        '''

        P = NeuronGroup(4000, eqs, threshold='v>Vt', reset='v = Vr',
                        refractory=5 * ms,
                        method='linear')
        P.v = 'Vr + rand() * (Vt - Vr)'

        we = (60 * 0.27 / 10) * mV  # excitatory synaptic weight (voltage)
        wi = (-20 * 4.5 / 10) * mV  # inhibitory synaptic weight
        Ce = Synapses(P, P, on_pre='ge += we')
        Ci = Synapses(P, P, on_pre='gi += wi')
        Ce.connect('i<3200', p=0.02)
        Ci.connect('i>=3200', p=0.02)

        s_mon = SpikeMonitor(P)

        run(1 * second)

    def time_COBAHH_example(self, target):
        # Parameters
        area = 20000 * umetre ** 2
        Cm = (1 * ufarad * cm ** -2) * area
        gl = (5e-5 * siemens * cm ** -2) * area

        El = -60 * mV
        EK = -90 * mV
        ENa = 50 * mV
        g_na = (100 * msiemens * cm ** -2) * area
        g_kd = (30 * msiemens * cm ** -2) * area
        VT = -63 * mV
        # Time constants
        taue = 5 * ms
        taui = 10 * ms
        # Reversal potentials
        Ee = 0 * mV
        Ei = -80 * mV
        we = 6 * nS  # excitatory synaptic weight
        wi = 67 * nS  # inhibitory synaptic weight

        # The model
        eqs = Equations('''
        dv/dt = (gl*(El-v)+ge*(Ee-v)+gi*(Ei-v)-
                 g_na*(m*m*m)*h*(v-ENa)-
                 g_kd*(n*n*n*n)*(v-EK))/Cm : volt
        dm/dt = alpha_m*(1-m)-beta_m*m : 1
        dn/dt = alpha_n*(1-n)-beta_n*n : 1
        dh/dt = alpha_h*(1-h)-beta_h*h : 1
        dge/dt = -ge*(1./taue) : siemens
        dgi/dt = -gi*(1./taui) : siemens
        alpha_m = 0.32*(mV**-1)*(13*mV-v+VT)/
                 (exp((13*mV-v+VT)/(4*mV))-1.)/ms : Hz
        beta_m = 0.28*(mV**-1)*(v-VT-40*mV)/
                (exp((v-VT-40*mV)/(5*mV))-1)/ms : Hz
        alpha_h = 0.128*exp((17*mV-v+VT)/(18*mV))/ms : Hz
        beta_h = 4./(1+exp((40*mV-v+VT)/(5*mV)))/ms : Hz
        alpha_n = 0.032*(mV**-1)*(15*mV-v+VT)/
                 (exp((15*mV-v+VT)/(5*mV))-1.)/ms : Hz
        beta_n = .5*exp((10*mV-v+VT)/(40*mV))/ms : Hz
        ''')

        P = NeuronGroup(4000, model=eqs, threshold='v>-20*mV',
                        refractory=3 * ms,
                        method='exponential_euler')
        Pe = P[:3200]
        Pi = P[3200:]
        Ce = Synapses(Pe, P, on_pre='ge+=we')
        Ci = Synapses(Pi, P, on_pre='gi+=wi')
        Ce.connect(p=0.02)
        Ci.connect(p=0.02)

        # Initialization
        P.v = 'El + (randn() * 5 - 5)*mV'
        P.ge = '(randn() * 1.5 + 4) * 10.*nS'
        P.gi = '(randn() * 12 + 20) * 10.*nS'

        # Record a few traces
        trace = StateMonitor(P, 'v', record=[1, 10, 100])
        run(1 * second, report='text')

    def time_STDP_example(self, target):
        N = 1000
        taum = 10 * ms
        taupre = 20 * ms
        taupost = taupre
        Ee = 0 * mV
        vt = -54 * mV
        vr = -60 * mV
        El = -74 * mV
        taue = 5 * ms
        F = 15 * Hz
        gmax = .01
        dApre = .01
        dApost = -dApre * taupre / taupost * 1.05
        dApost *= gmax
        dApre *= gmax

        eqs_neurons = '''
        dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
        dge/dt = -ge / taue : 1
        '''

        input = PoissonGroup(N, rates=F)
        neurons = NeuronGroup(1, eqs_neurons, threshold='v>vt', reset='v = vr',
                              method='linear')
        S = Synapses(input, neurons,
                     '''w : 1
                        dApre/dt = -Apre / taupre : 1 (event-driven)
                        dApost/dt = -Apost / taupost : 1 (event-driven)''',
                     on_pre='''ge += w
                            Apre += dApre
                            w = clip(w + Apost, 0, gmax)''',
                     on_post='''Apost += dApost
                             w = clip(w + Apre, 0, gmax)''',
                     )
        S.connect()
        S.w = 'rand() * gmax'
        mon = StateMonitor(S, 'w', record=[0, 1])
        s_mon = SpikeMonitor(input)

        run(10 * second, report='text')

FullExamples.params = ['numpy', 'cython', 'cpp_standalone']
FullExamples.param_names = ['target']
FullExamples.number = 1
FullExamples.timer = timeit.default_timer
FullExamples.timeout = 300


class Microbenchmarks():
    def setup(self, target):
        codegen_setup(target)

    def time_statemonitor(self, target):
        group = NeuronGroup(1000, 'v : 1')
        mon = StateMonitor(group, 'v', record=True)
        run(1*second)

    def peakmem_statemonitor(self, target):
        group = NeuronGroup(1000, 'v : 1')
        mon = StateMonitor(group, 'v', record=True)
        run(1*second)

Microbenchmarks.params = ['numpy', 'cython', 'cpp_standalone']
Microbenchmarks.param_names = ['target']
Microbenchmarks.number = 1
Microbenchmarks.timer = timeit.default_timer


class RuntimeBenchmarks():

    def time_repeated_run(self, target):
        taum = 20 * ms
        taue = 5 * ms
        taui = 10 * ms
        Vt = -50 * mV
        Vr = -60 * mV
        El = -49 * mV
        for _ in range(10):
            eqs = '''
                    dv/dt  = (ge+gi-(v-El))/taum : volt (unless refractory)
                    dge/dt = -ge/taue : volt
                    dgi/dt = -gi/taui : volt
                    '''
            P = NeuronGroup(4000, eqs, threshold='v>Vt', reset='v = Vr',
                            refractory=5 * ms, method='linear')
            P.v = 'Vr + rand() * (Vt - Vr)'

            we = (60 * 0.27 / 10) * mV  # excitatory synaptic weight (voltage)
            wi = (-20 * 4.5 / 10) * mV  # inhibitory synaptic weight
            Ce = Synapses(P, P, on_pre='ge += we')
            Ci = Synapses(P, P, on_pre='gi += wi')
            Ce.connect('i<3200', p=0.02)
            Ci.connect('i>=3200', p=0.02)
            spikes = SpikeMonitor(P)
            run(1*second)

    def peakmem_repeated_run(self, target):
        taum = 20 * ms
        taue = 5 * ms
        taui = 10 * ms
        Vt = -50 * mV
        Vr = -60 * mV
        El = -49 * mV
        for _ in range(10):
            eqs = '''
                    dv/dt  = (ge+gi-(v-El))/taum : volt (unless refractory)
                    dge/dt = -ge/taue : volt
                    dgi/dt = -gi/taui : volt
                    '''
            P = NeuronGroup(4000, eqs, threshold='v>Vt', reset='v = Vr',
                            refractory=5 * ms, method='linear')
            P.v = 'Vr + rand() * (Vt - Vr)'

            we = (60 * 0.27 / 10) * mV  # excitatory synaptic weight (voltage)
            wi = (-20 * 4.5 / 10) * mV  # inhibitory synaptic weight
            Ce = Synapses(P, P, on_pre='ge += we')
            Ci = Synapses(P, P, on_pre='gi += wi')
            Ce.connect('i<3200', p=0.02)
            Ci.connect('i>=3200', p=0.02)
            spikes = SpikeMonitor(P)
            run(1 * second)

    def time_repeated_run_store_restore(self, target):
        taum = 20 * ms
        taue = 5 * ms
        taui = 10 * ms
        Vt = -50 * mV
        Vr = -60 * mV
        El = -49 * mV
        eqs = '''
                dv/dt  = (ge+gi-(v-El))/taum : volt (unless refractory)
                dge/dt = -ge/taue : volt
                dgi/dt = -gi/taui : volt
              '''
        P = NeuronGroup(4000, eqs, threshold='v>Vt', reset='v = Vr',
                        refractory=5 * ms, method='linear')
        P.v = 'Vr + rand() * (Vt - Vr)'

        we = (60 * 0.27 / 10) * mV  # excitatory synaptic weight (voltage)
        wi = (-20 * 4.5 / 10) * mV  # inhibitory synaptic weight
        Ce = Synapses(P, P, on_pre='ge += we')
        Ci = Synapses(P, P, on_pre='gi += wi')
        spikes = SpikeMonitor(P)
        store()
        for _ in range(10):
            restore()
            Ce.connect('i<3200', p=0.02)
            Ci.connect('i>=3200', p=0.02)
            run(1*second)

RuntimeBenchmarks.params = ['numpy', 'cython']
RuntimeBenchmarks.param_names = ['target']
RuntimeBenchmarks.timeout = 300


class UnitBenchmarks():
    def setup(self):
        self.ar = np.random.randn(1000000)
        self.ar_with_units = self.ar * mV

    def time_calculations_with_units(self):
        rmse = np.sqrt(np.mean(self.ar_with_units**2))

    def time_units_to_string(self):
        str(self.ar_with_units)
