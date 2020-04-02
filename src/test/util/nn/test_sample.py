import pytest

import src.main.util.nn.sample as sample


def test_get_samples_order_independent_net(SNAPTwitter_nodes, SNAPTwitter_emb,
                                           SNAPTwitter_p_gins):
    samples = sample.get_samples_order_independent_net(SNAPTwitter_nodes, SNAPTwitter_emb,
                                                       SNAPTwitter_p_gins)
    assert samples.size() == (47278, 257)

def test_split_samples(samples):
    training_set, validation_set = sample.split_samples(samples)
    assert training_set.size() == (8, 3)