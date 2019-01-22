from delira.data_loading.sampler import LambdaSampler, \
                                        PrevalenceRandomSampler, \
                                        PrevalenceSequentialSampler, \
                                        RandomSampler, \
                                        SequentialSampler, \
                                        StoppingPrevalenceRandomSampler, \
                                        StoppingPrevalenceSequentialSampler, \
                                        WeightedRandomSampler

import numpy as np
from . import DummyDataset

def test_lambda_sampler():
    np.random.seed(1)
    dset = DummyDataset(600, [0.5, 0.3, 0.2])

    def sampling_fn_a(index_list, n_indices):
        return index_list[:n_indices]

    def sampling_fn_b(index_list, n_indices):
        return index_list[-n_indices:]
    
    sampler_a = LambdaSampler(list(range(len(dset))), sampling_fn_a)
    sampler_b = LambdaSampler(list(range(len(dset))), sampling_fn_b)

    assert sampler_a(15) == list(range(15))
    assert sampler_b(15) == list(range(len(dset) - 15, len(dset)))

def test_prevalence_random_sampler():
    np.random.seed(1)
    dset = DummyDataset(600, [0.5, 0.3, 0.2])
    
    sampler = PrevalenceRandomSampler.from_dataset(dset)

    for batch_len in [1, 2, 3]:
    
        equal_batch = sampler(batch_len)

        seen_labels = []
        for idx in equal_batch:
            curr_label = dset[idx]["label"]

            if curr_label not in seen_labels:
                seen_labels.append(curr_label)
            else:
                assert False, "Label already seen and labels must be unique. \
                                Batch length: %d" % batch_len


    assert len(sampler(5)) == 5

def test_prevalence_sequential_sampler():
    np.random.seed(1)
    dset = DummyDataset(600, [0.5, 0.3, 0.2])
    
    sampler = PrevalenceSequentialSampler.from_dataset(dset)

    # ToDo add test considering actual sampling strategy

    assert len(sampler(5)) == 5

def test_random_sampler():
    np.random.seed(1)
    dset = DummyDataset(600, [0.5, 0.3, 0.2])

    sampler = RandomSampler.from_dataset(dset)
    
    assert len(sampler(250)) == 250

    # checks if labels are all the same (should not happen if random sampled)
    assert len(set([dset[_idx]["label"] for _idx in sampler(301)])) > 1

def test_sequential_sampler():
    np.random.seed(1)
    dset = DummyDataset(600, [0.5, 0.3, 0.2])

    sampler = SequentialSampler.from_dataset(dset)

    # if sequentially sampled, the first 300 items should have label 0 -> 1 
    # unique element
    assert len(set([dset[_idx]["label"] for _idx in sampler(100)])) == 1
    assert len(sampler(100)) == 100
    # next 100 elements also same label -> next 201 elements: two different 
    # labels
    assert len(set([dset[_idx]["label"] for _idx in sampler(101)])) == 2

def test_stopping_prevalence_random_sampler():
    np.random.seed(1)
    dset = DummyDataset(600, [0.5, 0.3, 0.2])

    sampler = StoppingPrevalenceRandomSampler.from_dataset(dset)

    try:
        for i in range(121):
            sample = sampler(3)
            assert len(set([dset[_idx]["label"] for _idx in sample])) == 3

        assert False, "Sampler should have raised StopIteration by now"

    except StopIteration:
        assert True

def test_stopping_prevalence_sequential_sampler():
    np.random.seed(1)
    dset = DummyDataset(600, [0.5, 0.3, 0.2])

    sampler = StoppingPrevalenceRandomSampler.from_dataset(dset)

    try:
        for i in range(121):
            sample = sampler(3)
            assert len(set([dset[_idx]["label"] for _idx in sample])) == 3

        assert False, "Sampler should have raised StopIteration by now"

    except StopIteration:
        assert True


if __name__ == '__main__':
    test_lambda_sampler()
    test_prevalence_random_sampler()
    test_prevalence_sequential_sampler()
    test_random_sampler()
    test_sequential_sampler()
    test_stopping_prevalence_random_sampler()
    test_stopping_prevalence_sequential_sampler()