from collections import UserDict

import pytest

from llm_platform.tokenization import single_sequence_token_ids


@pytest.mark.parametrize("encoded", [
    [7, 11, 13], (7, 11, 13), [[7, 11, 13]],
    {"input_ids": [7, 11, 13], "attention_mask": [1, 1, 1]},
    UserDict(input_ids=[[7, 11, 13]], attention_mask=[[1, 1, 1]]),
])
def test_single_conversation_counts_ids_instead_of_mapping_keys_or_batch_rows(encoded):
    assert single_sequence_token_ids(encoded) == [7, 11, 13]


@pytest.mark.parametrize("encoded", [
    [], [[]], [[1, 2], [3, 4]], [[[1, 2]]], {"attention_mask": [1]},
    {"input_ids": [[1], [2]]}, [True], [1.0], [-1], "1,2", 1, None,
])
def test_malformed_or_multiple_conversations_cannot_bypass_token_limits(encoded):
    with pytest.raises(ValueError, match="Tokenization"):
        single_sequence_token_ids(encoded)


def test_real_cpu_tensor_flat_singleton_and_malformed_shapes():
    torch = pytest.importorskip("torch")
    assert single_sequence_token_ids(torch.tensor([3, 5])) == [3, 5]
    assert single_sequence_token_ids({"input_ids": torch.tensor([[3, 5]])}) == [3, 5]
    for tensor in (torch.tensor([[3], [5]]), torch.tensor([[[3, 5]]]),
                   torch.tensor([True]), torch.tensor([3.0])):
        with pytest.raises(ValueError, match="Tokenization"):
            single_sequence_token_ids(tensor)
