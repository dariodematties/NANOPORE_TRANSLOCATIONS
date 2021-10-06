import torch
import numpy as np

class Artificial_DataLoader:
    def __init__(self, world_size, rank, device, File, sampling_rate, number_of_concentrations, number_of_durations, number_of_diameters, window, length, batch_size,
    max_num_of_pulses_in_a_wind=75):
        assert window < length

        self.world_size = world_size
        self.rank = rank
        self.rank_id = rank    # the rank id will change according to the epoch
        self.device = device

        self.File = File

        self.sampling_rate = sampling_rate
        self.number_of_concentrations = number_of_concentrations
        self.number_of_durations = number_of_durations
        self.number_of_diameters = number_of_diameters
        self.windows_per_signal = int(length / window)

        # this is the shape of the structure of windows in the dataset
        self.shape = (number_of_concentrations, number_of_durations, number_of_diameters, int(length / window))

        # this is the total number of windows in the dataset
        self.total_number_of_windows = self.number_of_concentrations * \
                                       self.number_of_durations * \
                                       self.number_of_diameters * \
                                       self.windows_per_signal

        # this is the size of the fragment from the total number of windows that corresponds to this rank
        self.shard_size = self.total_number_of_windows // world_size
        # if there is residue in the distribution of windows among ranks
        # all shard sizes have to be incremented in one
        # since all shard sizes have to be equal
        if self.total_number_of_windows % world_size != 0:
            self.shard_size += 1

        self.window = window
        self.length = length
        self.batch_size = batch_size
        self.max_num_of_pulses_in_a_wind = max_num_of_pulses_in_a_wind
        self.avail_winds = self.get_avail_winds(self.shard_size)

        # unravel indices in advance to avoid computational cost during execution
        auxiliary = [i for i in range(self.total_number_of_windows)]
        self.unraveled_indices = np.unravel_index(auxiliary, self.shape)
        
        self.samples_indices = []
        self.number_of_avail_windows = self.get_number_of_avail_windows()
 

    @staticmethod
    def get_avail_winds(shard_size):
        return torch.ones((shard_size), dtype=bool)


    # determines the quota of any number of things among ranks including residues
    # for instance if total is 100 and world_size is 3, then rank 0 will have a quota of 4
    # rank 1 a quota of 3 and rank 2 a quota of 3 too.
    def _get_quota(self, world_size, rank, total):
        assert(total >= world_size)
        quota = total // world_size
        residue = total % world_size
        if rank < residue:
            quota += 1

        return quota
 

    # restart all the available windows as it is when the object is created
    # it rotates the identity of ranks at each epoch in order to make each rank to "see" all the samples
    def reset_avail_winds(self, epoch):
        self.rank_id = (self.rank + epoch) % self.world_size

        # this is the fragment from the total number of windows that corresponds to this rank
        self.shard_size = self._get_quota(self.world_size, self.rank_id, self.total_number_of_windows)

        self.avail_winds = torch.ones((self.shard_size), dtype=bool)
        
        self.number_of_avail_windows = self.get_number_of_avail_windows()
 

    # make 100 random windows available
    def _reset_random_winds(self):
        i = 0
        while i < 100:
            window = torch.randint(0, self.shard_size, (1,1))[0].item()
            if self.avail_winds[window] == False:
                self.avail_winds[window] = True
                i += 1
        
        self.number_of_avail_windows += 100

    # returns the number of available windows in the object
    def get_number_of_avail_windows(self):
        return sum(self.avail_winds==True).item()

    # map from the local available resource in the rank to the global available resource in the world
    # For instance, global resource is:
    # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    # world_size = 3
    #
    # rank 0                rank 1                  rank 2
    # 0, 1, 2, 3            0, 1, 2                 0, 1, 2
    #
    # the mapping formula is:
    # (sample * world_size) + rank
    #
    # rank 0                rank 1                  rank 2
    # (0 * 3) + 0 = 0       (0 * 3) + 1 = 1         (0 * 3) + 2 = 2
    # (1 * 3) + 0 = 3       (1 * 3) + 1 = 4         (1 * 3) + 2 = 5
    # (2 * 3) + 0 = 6       (2 * 3) + 1 = 7         (2 * 3) + 2 = 8
    # (3 * 3) + 0 = 9
    #
    # this situation is going to rotate in according to the epoch
    def _map_from_rank_to_world(self, sample):
        return (sample * self.world_size) + self.rank_id

    # get a sample from the available windows and set the sample as unavailable
    def _get_sample(self):
        if (self.number_of_avail_windows == 0):
            self._reset_random_winds()

        availables = np.where(self.avail_winds)
        idx = torch.randint(0, availables[0].size, (1,1))[0].item()
        sampled_window = availables[0][idx]
        
        # set window as unavailable
        self.avail_winds[sampled_window] = False

        # map the sample from the rank domain to the global resources
        sampled_window = self._map_from_rank_to_world(sampled_window)
        sampled_window = (self.unraveled_indices[0][sampled_window], \
                          self.unraveled_indices[1][sampled_window], \
                          self.unraveled_indices[2][sampled_window], \
                          self.unraveled_indices[3][sampled_window],)

        #sampled_window = np.unravel_index(sampled_window, self.shape)
        self.number_of_avail_windows -= 1
        return sampled_window


    def _get_labels(self, time_window, Cnp, Duration, Dnp):
        "Returns classes and bboxes inside the signal window"
        dset_p = self.File['Cnp_' + str(Cnp+1) + '/Duration_' + str(Duration+1) + '/Dnp_' + str(Dnp+1) + '/parameters']
        pulses_inside_window = np.where((torch.from_numpy(dset_p[0,:]) > time_window[0].cpu()) & \
                                        (torch.from_numpy(dset_p[0,:]) < time_window[-1].cpu()))[0]
        pulses_inside_window = pulses_inside_window.tolist()
        
        start_times = dset_p[0,pulses_inside_window]
        #pulse_widths = dset_p[1,pulses_inside_window]
        #pulse_categories = dset_p[2,pulses_inside_window]
        pulse_widths = dset_p[2,pulses_inside_window]
        pulse_amplitudes = dset_p[3,pulses_inside_window]
        
        number_of_pulses = len(pulses_inside_window)
        if number_of_pulses == 0:
            average_width = 0.0
            average_amplitude = 0.0
        else:
            average_width = np.average(pulse_widths)
            average_amplitude = np.average(pulse_amplitudes)

        starts = (torch.from_numpy(start_times) - time_window[0].cpu()) / self.window
        widths = torch.from_numpy(pulse_widths) / self.window
        amplitudes = torch.from_numpy(pulse_amplitudes)
        
        starts = starts.tolist()
        widths = widths.tolist()
        amplitudes = amplitudes.tolist()
        #categories = pulse_categories.tolist()
        categories = np.zeros(len(pulses_inside_window)).tolist()
        
        starts = (starts + [1.0]*(self.max_num_of_pulses_in_a_wind - len(starts)))
        widths = (widths + [1.0]*(self.max_num_of_pulses_in_a_wind - len(widths)))
        amplitudes = (amplitudes + [1.0]*(self.max_num_of_pulses_in_a_wind - len(amplitudes)))
        categories = (categories + [1.0]*(self.max_num_of_pulses_in_a_wind - len(categories)))
        
        starts = torch.FloatTensor(starts)
        widths = torch.FloatTensor(widths)
        amplitudes = torch.FloatTensor(amplitudes)
        categories = torch.FloatTensor(categories)
        return starts, widths, amplitudes, categories, number_of_pulses, average_width, average_amplitude


    def _get_signal_window(self, with_labels=False):
        if len(self.samples_indices) == 0: # bring 100 samples
            for i in range(100):
                self.samples_indices.append(self._get_sample())

        sample = self.samples_indices.pop(0)
        Cnp = sample[0]
        Duration = sample[1]
        Dnp = sample[2]
        window_number = sample[3]
        dset = self.File['Cnp_' + str(Cnp+1) + '/Duration_' + str(Duration+1) + '/Dnp_' + str(Dnp+1) + '/data']
        #assert dset.shape[1] % self.length == 0
        samples_per_second = int(dset.shape[1] / self.length)
        samples_per_window = int(samples_per_second * self.window)
        begin = window_number * samples_per_window
        end = begin + samples_per_window
        time_window = torch.Tensor(dset[0,begin:end]).to(self.device)
        clean_signal = torch.Tensor(dset[1,begin:end]).to(self.device)
        noisy_signal = torch.Tensor(dset[2,begin:end]).to(self.device)

        if with_labels:
            starts, widths, amplitudes, categories, number_of_pulses, average_width, average_amplitude = self._get_labels(time_window, Cnp, Duration, Dnp)
            return time_window, clean_signal, noisy_signal, starts, widths, amplitudes, categories, number_of_pulses, average_width, average_amplitude
        else:
            return time_window, clean_signal, noisy_signal

        
    def get_batch(self, descart_empty_windows=True):
        #assert sum(self.avail_winds == True) > self.batch_size

        noisy_signals = torch.Tensor(self.batch_size, int(self.window*self.sampling_rate)).to(self.device)
        clean_signals = torch.Tensor(self.batch_size, int(self.window*self.sampling_rate)).to(self.device)
        times = torch.Tensor(self.batch_size, int(self.window*self.sampling_rate)).to(self.device)
        pulse_labels = torch.Tensor(self.batch_size, 4, self.max_num_of_pulses_in_a_wind).to(self.device)
        average_labels = torch.Tensor(self.batch_size, 3).to(self.device)
        for i in range(self.batch_size):
            number_of_pulses = 0
            if descart_empty_windows:
                while(number_of_pulses==0):
                    Time, Clean_signal, Noisy_signal, starts, widths, amplitudes, categories,\
                            number_of_pulses, average_width, average_amplitude = self._get_signal_window(with_labels=True)
            else:
                Time, Clean_signal, Noisy_signal, starts, widths, amplitudes, categories,\
                        number_of_pulses, average_width, average_amplitude = self._get_signal_window(with_labels=True)

            times[i] = Time
            clean_signals[i] = Clean_signal
            noisy_signals[i] = Noisy_signal
            pulse_labels[i][0] = starts
            pulse_labels[i][1] = widths
            pulse_labels[i][2] = amplitudes
            pulse_labels[i][3] = categories
            
            average_labels[i][0] = number_of_pulses
            average_labels[i][1] = average_width
            average_labels[i][2] = average_amplitude

        return times, noisy_signals, clean_signals, pulse_labels, average_labels






    def get_signal_window(self, Cnp, Duration, Dnp, window_number):
        dset = self.File['Cnp_' + str(Cnp+1) + '/Duration_' + str(Duration+1) + '/Dnp_' + str(Dnp+1) + '/data']
        #assert dset.shape[1] % self.length == 0
        samples_per_second = int(dset.shape[1] / self.length)
        samples_per_window = int(samples_per_second * self.window)
        begin = window_number * samples_per_window
        end = begin + samples_per_window
        time_window = torch.Tensor(dset[0,begin:end]).to(self.device)
        clean_signal = torch.Tensor(dset[1,begin:end]).to(self.device)
        noisy_signal = torch.Tensor(dset[2,begin:end]).to(self.device)

        starts, widths, amplitudes, categories, number_of_pulses, average_width, average_amplitude = self._get_labels(time_window, Cnp, Duration, Dnp)
        pulse_labels = torch.Tensor(4, self.max_num_of_pulses_in_a_wind).to(self.device)
        average_labels = torch.Tensor(3).to(self.device)

        pulse_labels[0] = starts
        pulse_labels[1] = widths
        pulse_labels[2] = amplitudes
        pulse_labels[3] = categories

        average_labels[0] = number_of_pulses
        average_labels[1] = average_width
        average_labels[2] = average_amplitude

        return time_window, noisy_signal, clean_signal, pulse_labels, average_labels









class Unlabeled_Real_DataLoader:
    def __init__(self, device, File, num_of_traces, window, length):
        assert window < length

        self.device = device

        self.File = File

        self.num_of_traces = num_of_traces
        self.windows_per_trace = int(length / window)

        # this is the shape of the structure of windows in the dataset
        self.shape = (self.num_of_traces, int(length / window))

        # this is the total number of windows in the dataset
        self.total_number_of_windows = self.num_of_traces * self.windows_per_trace

        self.window = window
        self.length = length


    def get_signal_window(self, trace_number, window_number):
        dset = self.File['Volt_' + str(trace_number+1) + '/data']
        #assert dset.shape[1] % self.length == 0
        samples_per_second = int(dset.shape[1] / self.length)
        samples_per_window = int(samples_per_second * self.window)
        begin = window_number * samples_per_window
        end = begin + samples_per_window
        time_window = torch.Tensor(dset[0,begin:end]).to(self.device)
        signal = torch.Tensor(dset[1,begin:end]).to(self.device)

        return time_window, signal






class Labeled_Real_DataLoader:
    def __init__(self, device, File, num_of_traces, window, length):
        assert window < length

        self.device = device

        self.File = File

        self.num_of_traces = num_of_traces
        self.windows_per_trace = int(length / window)

        # this is the shape of the structure of windows in the dataset
        self.shape = (self.num_of_traces, int(length / window))

        # this is the total number of windows in the dataset
        self.total_number_of_windows = self.num_of_traces * self.windows_per_trace

        self.window = window
        self.length = length


    def _get_labels(self, time_window, trace_number):
        "Returns classes and bboxes inside the signal window"
        dset_p = self.File['Volt_' + str(trace_number+1) + '/parameters']
        pulses_inside_window = np.where((torch.from_numpy(dset_p[0,:]) > time_window[0].cpu()) & \
                                        (torch.from_numpy(dset_p[0,:]) < time_window[-1].cpu()))[0]
        pulses_inside_window = pulses_inside_window.tolist()
        
        start_times = dset_p[0,pulses_inside_window]
        #pulse_widths = dset_p[1,pulses_inside_window]
        #pulse_categories = dset_p[2,pulses_inside_window]
        pulse_widths = dset_p[2,pulses_inside_window]
        pulse_amplitudes = dset_p[3,pulses_inside_window]
        
        number_of_pulses = len(pulses_inside_window)
        if number_of_pulses == 0:
            average_width = 0.0
            average_amplitude = 0.0
        else:
            average_width = np.average(pulse_widths)
            average_amplitude = np.average(pulse_amplitudes)

        starts = (torch.from_numpy(start_times) - time_window[0].cpu()) / self.window
        widths = torch.from_numpy(pulse_widths) / self.window
        amplitudes = torch.from_numpy(pulse_amplitudes)
        
        return starts, widths, amplitudes, number_of_pulses, average_width, average_amplitude



    def get_signal_window(self, trace_number, window_number):
        dset = self.File['Volt_' + str(trace_number+1) + '/data']
        #assert dset.shape[1] % self.length == 0
        samples_per_second = int(dset.shape[1] / self.length)
        samples_per_window = int(samples_per_second * self.window)
        begin = window_number * samples_per_window
        end = begin + samples_per_window
        time_window = torch.Tensor(dset[0,begin:end]).to(self.device)
        noisy_signal = torch.Tensor(dset[1,begin:end]).to(self.device)

        starts, widths, amplitudes, number_of_pulses, average_width, average_amplitude = self._get_labels(time_window, trace_number)
        num_of_pulses_in_the_wind = starts.shape[0]
        pulse_labels = torch.Tensor(3, num_of_pulses_in_the_wind).to(self.device)
        average_labels = torch.Tensor(3).to(self.device)

        pulse_labels[0] = starts
        pulse_labels[1] = widths
        pulse_labels[2] = amplitudes

        average_labels[0] = number_of_pulses
        average_labels[1] = average_width
        average_labels[2] = average_amplitude

        return time_window, noisy_signal, pulse_labels, average_labels
