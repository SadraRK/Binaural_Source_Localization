import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore
from scipy.io import savemat, loadmat

######################################################################## Neural Network ########################################################################
class NeuralNet:
    def __init__(self):
        self.Vthresh = 1
        self.Cmem = 1e-12  # 1pF
        self.Erev = 2.0
        self.tau_decay = 10
        self.tau_learn = 1
        self.dt = 1e-6     # 1us step size
        self.Gleak = 1e-8
        self.Eleak = 0.0

    def Fire(self, I_in, t, Gsyn, lrn_rate):
        tspike_pre = np.zeros((len(t), 2)) # Initialize pre-synaptic spike times
        tspike_post = np.zeros((len(t)))     # Initialize post-synaptic spike times
        Isyn = np.zeros((len(t)+1, 3)) # Initialize synaptic currents
        Vmem = np.zeros((len(t), 3)) # Initialize membrane potentials
        W = Gsyn
        deltaG0 = 0
        deltaG1 = 0

        for tnow in range(1, len(t)):
            Vmem[tnow, 0] = self.calc_Vmem(Vmem[tnow - 1, 0], I_in[0, tnow], self.Cmem, self.dt)
            Vmem[tnow, 1] = self.calc_Vmem(Vmem[tnow - 1, 1], I_in[1, tnow], self.Cmem, self.dt)
            idx = np.where(Vmem[tnow] > self.Vthresh)
            if np.size(idx[0])>0:
                Vmem[tnow, idx] = 0.0
                tspike_pre[tnow, idx] = 1
                pre_idx = np.where(tspike_pre>= 1)
                if ((np.size(pre_idx[0][pre_idx[1]==0])>0) and (tnow >= pre_idx[0][-1])):
                    Isyn[:, 0] = self.calc_Isyn(Isyn[:, 0], Gsyn[0], pre_idx[0][pre_idx[1]==0][-1], tnow, self.tau_decay, self.Erev, Vmem[:, 0])
                if ((np.size(pre_idx[0][pre_idx[1] == 1]) > 0) and (tnow >= pre_idx[0][-1])):
                    Isyn[:, 1] = self.calc_Isyn(Isyn[:, 1], Gsyn[1], pre_idx[0][pre_idx[1]==1][-1], tnow, self.tau_decay, self.Erev, Vmem[:, 1])
            Vmem[tnow, 2] = self.calc_Vmem(Vmem[tnow - 1, 2], (Isyn[tnow, 0] + Isyn[tnow, 1]), self.Cmem, self.dt)
            if Vmem[tnow, 2] > self.Vthresh:
                Vmem[tnow, 2] = 0.0
                tspike_post[tnow] = 1
                t_tspike_post = tnow
                left_spike = tspike_pre[tnow, 0]
                right_spike = tspike_pre[tnow, 1]
                time_delay = left_spike - right_spike
                # if time_delay < 0:
                #     deltaG0 -= lrn_rate*self.tau_learn/10
                # elif time_delay > 0:
                #     deltaG1 -= lrn_rate*self.tau_learn*4/7
            idx_post = np.where(tspike_post == 1)[0]
            if (idx_post.size > 0) and (lrn_rate != 0):
                deltaG0 += np.sum(lrn_rate*tnow* np.exp(-(tnow-tspike_post[idx_post[-1]])/self.tau_learn)*Vmem[:, 0])
                deltaG1 += np.sum(lrn_rate*tnow* np.exp(-(tnow-tspike_post[idx_post[-1]])/self.tau_learn)*Vmem[:, 1])
            W = Gsyn + np.array([deltaG0, deltaG1]) * self.dt
        pre_Synaptic_time = np.where(tspike_pre>= 1)
        post_Synaptic_time = np.where(tspike_post >= 1)
        # print(tspike_pre[pre_Synaptic_time[0][pre_Synaptic_time[1]==0]])
        return Isyn, Vmem, W, tspike_pre, tspike_post

    def train(self, I_in, t, t_Gsyn, lrn_rate, epoch):
        tempW = t_Gsyn
        for i in range(epoch):
            temp_Isyn, temp_Vmem, tempW, pre_Synaptic_time, post_Synaptic_time = self.Fire(I_in, t, tempW, lrn_rate)
            print("Training Epoch #" + str(i+1))
        return temp_Isyn, temp_Vmem, tempW

    def test(self, I_in, t, t_Gsyn):
        return self.Fire(I_in, t, t_Gsyn, 0)

    def calc_Isyn(self, Isyn, Gsyn, tspike_pre, tnow, tau_decay, Erev, Vmem):
        Isyn[tnow + 1] += Gsyn * np.exp((tspike_pre - tnow) / tau_decay) * (Erev - Vmem[tnow])
        return Isyn

    def calc_Vmem(self, Vm_init, Isyn, Cmem, dt):
        return Vm_init + np.sum(Isyn, axis=0) * dt / Cmem + self.Gleak * (self.Eleak - Vm_init) * dt / Cmem


######################################################################## Run Binaural Audio Localization ########################################################################

if __name__ == '__main__':

    #################################################################################################################### Generating Current Pulses
    t = np.linspace(0, 15, 360000)
    I0 = 1e-6
    I1 = 1e-6
    t_pulse1 = 1000  # 1ms pulse width for I0
    t_pulse2 = 1000 # 1ms pulse width for I1
    t_start_1 = 1000
    t_start_2 = 6000
    FR_mid    = 5919442
    FR_lower  = 3995867 # 3995867
    FR_higher = 4328759 # 4328759
    db_path = 'Binaural_Dataset'
    audio = loadmat(db_path + '/Processed/sub_1_2.mat')
    audio = audio['audio']
    audio_left = audio[0, :]
    audio_right = audio[1, :]

    spike_times_left = np.where(audio_left > 0.1)[0]
    spike_times_right = np.where(audio_right > 0.1)[0]
    spike_train_left = np.zeros((len(audio_left)))
    spike_train_right_0deg = np.zeros((len(audio_right)))
    spike_train_right_45deg = np.zeros((len(audio_right)))
    spike_train_right_70deg = np.zeros((len(audio_right)))
    spike_train_right_180deg = np.zeros((len(audio_right)))
    spike_train_right = np.zeros((len(audio_right)))
    start_offset = spike_times_left[0] - spike_times_right[0]
    spike_train_left[spike_times_left] = audio_left[spike_times_left]
    spike_train_right_0deg[spike_times_right+start_offset+14] = audio_right[spike_times_right]
    spike_train_right_45deg[spike_times_right + start_offset + 12] = audio_right[spike_times_right]
    spike_train_right_70deg[spike_times_right + start_offset + 10] = audio_right[spike_times_right]
    spike_train_right_180deg[spike_times_right+start_offset - 14] = audio_right[spike_times_right]
    spike_train_right[spike_times_right] = audio_right[spike_times_right]

    I0_Wave_init = I0 * spike_train_left
    I1_Wave_init_0deg = I0 * spike_train_right_0deg
    I1_Wave_init_45deg = I0 * spike_train_right_45deg
    I1_Wave_init_170deg = I0 * spike_train_right_70deg
    I1_Wave_init_180deg = I0 * spike_train_right_180deg
    I1_Wave_init = I1 * spike_train_right

    #################################################################################################################### Initialization Phase
    Gsyn = np.array([2e-8, 2e-8])
    SNN = NeuralNet()

    initial_Isyn, initial_Vmem, initial_Gsyn, initial_pre_Synaptic_time, initial_post_Synaptic_time = SNN.test(np.array([I0_Wave_init, I1_Wave_init_0deg]), t, Gsyn)
    FR = np.sum(np.where(initial_post_Synaptic_time == 1)[0])
    print(Fore.WHITE + "Initialization Done!")
    print(Fore.GREEN + "Initial synaptic weights: Gsyn0=", initial_Gsyn[0], ", Gsyn1=", initial_Gsyn[1])
    print(Fore.BLUE + "Initial Firing Rate: ", FR)

    #################################################################################################################### Training Phase
    train_Isyn, train_Vmem, train_Gsyn = SNN.train(np.array([I0_Wave_init, I1_Wave_init_0deg]), t, initial_Gsyn, lrn_rate= 0.001, epoch= 5)
    print(Fore.WHITE + "Training Done!")
    print(Fore.GREEN + "Final synaptic weights: Gsyn0=", train_Gsyn[0], ", Gsyn1=", train_Gsyn[1])

    #################################################################################################################### Testing Phase
    test_Isyn, test_Vmem, test_Gsyn, test_pre_Synaptic_time, test_post_Synaptic_time = SNN.test(np.array([I0_Wave_init, I1_Wave_init_0deg]), t, train_Gsyn)
    FR = np.sum(np.where(test_post_Synaptic_time == 1)[0])
    angle = abs(FR-FR_lower)/abs(FR_higher-FR_lower) * 180
    print(Fore.WHITE + "Testing Done!")
    print(Fore.BLUE + "Final Firing Rate: ", FR)
    print(Fore.MAGENTA + "Azimuth Angle: ", angle)

    #################################################################################################################### Plotting
    plt.plot(t, I0_Wave_init*1e6, t, I1_Wave_init*1e6)
    plt.xlabel("Time(ms)")
    plt.ylabel("Amplitude (uA)")
    plt.legend(['I0', 'I1'])
    plt.title('Initial Phase')
    plt.show()
    plt.plot(t, initial_Vmem[:, 0], t, initial_Vmem[:, 1])
    plt.xlabel("Time(ms)")
    plt.ylabel("Amplitude (V)")
    plt.legend(['Vmem0', 'Vmem1'])
    plt.title('Initial Phase')
    plt.show()
    plt.plot(t, initial_Isyn[0:-1, 0], t, initial_Isyn[0:-1, 1])
    plt.xlabel("Time(ms)")
    plt.ylabel("Current (A)")
    plt.legend(['Isyn0', 'Isyn1'])
    plt.title('Initial Phase')
    plt.show()
    plt.plot(t, initial_Vmem[:, 2])
    plt.xlabel("Time(ms)")
    plt.ylabel("Amplitude (V)")
    plt.legend(['Vmem_Out'])
    plt.title('Initial Phase')
    plt.show()
    plt.subplot(3, 1, 1)
    plt.plot(t, initial_pre_Synaptic_time[:, 0])
    plt.xlabel("Time(ms)")
    plt.ylabel("Amplitude (a.u.)")
    plt.legend(['N0 Spike Timing'])
    plt.subplot(3, 1, 2)
    plt.plot(t, initial_pre_Synaptic_time[:, 1])
    plt.xlabel("Time(ms)")
    plt.ylabel("Amplitude (a.u.)")
    plt.legend(['N1 Spike Timing'])
    plt.subplot(3, 1, 3)
    plt.plot(t, initial_post_Synaptic_time[:])
    plt.xlabel("Time(ms)")
    plt.ylabel("Amplitude (a.u.)")
    plt.legend(['N_Out Spike Timing'])
    plt.show()

    plt.plot(t, I0_Wave_init, t, I1_Wave_init)
    plt.xlabel("Time(ms)")
    plt.ylabel("Amplitude (uA)")
    plt.legend(['I0', 'I1'])
    plt.title('Training Phase')
    plt.show()
    plt.plot(t, train_Vmem[:, 0], t, train_Vmem[:, 1])
    plt.xlabel("Time(ms)")
    plt.ylabel("Amplitude (V)")
    plt.legend(['Vmem0', 'Vmem1'])
    plt.title('Training Phase')
    plt.show()
    plt.plot(t, train_Isyn[0:-1, 0], t, train_Isyn[0:-1, 1])
    plt.xlabel("Time(ms)")
    plt.ylabel("Current (A)")
    plt.legend(['Isyn0', 'Isyn1'])
    plt.title('Training Phase')
    plt.show()
    plt.plot(t, train_Vmem[:, 2])
    plt.xlabel("Time(ms)")
    plt.ylabel("Amplitude (V)")
    plt.legend(['Vmem_Out'])
    plt.title('Training Phase')
    plt.show()

    plt.plot(t, I0_Wave_init*1e6, t, I1_Wave_init*1e6)
    plt.xlabel("Time(ms)")
    plt.ylabel("Amplitude (uA)")
    plt.legend(['I0', 'I1'])
    plt.title('Test Phase')
    plt.show()
    plt.plot(t, test_Vmem[:, 0], t, test_Vmem[:, 1])
    plt.xlabel("Time(ms)")
    plt.ylabel("Amplitude (V)")
    plt.legend(['Vmem0', 'Vmem1'])
    plt.title('Test Phase')
    plt.show()
    plt.plot(t*3, test_Isyn[0:-1, 0], t, test_Isyn[0:-1, 1])
    plt.xlabel("Time(ms)")
    plt.ylabel("Current (A)")
    plt.legend(['Isyn0', 'Isyn1'])
    plt.title('Test Phase')
    plt.show()
    plt.plot(t, test_Vmem[:, 2])
    plt.xlabel("Time(ms)")
    plt.ylabel("Amplitude (V)")
    plt.legend(['Vmem_Out'])
    plt.title('Test Phase')
    plt.show()
    plt.subplot(3, 1, 1)
    plt.plot(t, test_pre_Synaptic_time[:, 0])
    plt.xlabel("Time(ms)")
    plt.ylabel("Amplitude (a.u.)")
    plt.legend(['N0 Spike Timing'])
    plt.subplot(3, 1, 2)
    plt.plot(t, test_pre_Synaptic_time[:, 1])
    plt.xlabel("Time(ms)")
    plt.ylabel("Amplitude (a.u.)")
    plt.legend(['N1 Spike Timing'])
    plt.subplot(3, 1, 3)
    plt.plot(t, test_post_Synaptic_time[:])
    plt.xlabel("Time(ms)")
    plt.ylabel("Amplitude (a.u.)")
    plt.legend(['N_Out Spike Timing'])
    plt.show()