import torch

from recsys.models.two_tower import TwoTower


def make_model() -> TwoTower:
    torch.manual_seed(0)
    return TwoTower(num_users=50, num_items=30, embed_dim=16, hidden_dim=32, out_dim=8)


def test_towers_output_normalized_embeddings():
    model = make_model()
    u, v = model(torch.arange(10), torch.arange(10))
    assert u.shape == (10, 8) and v.shape == (10, 8)
    assert torch.allclose(u.norm(dim=-1), torch.ones(10), atol=1e-5)
    assert torch.allclose(v.norm(dim=-1), torch.ones(10), atol=1e-5)


def test_in_batch_loss_decreases_with_training():
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    # Fixed positive pairs: user i likes item i % 30.
    users = torch.arange(50)
    items = users % 30
    initial = model.in_batch_loss(users, items).item()
    for _ in range(30):
        loss = model.in_batch_loss(users, items)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < initial


def test_encode_items_covers_catalogue():
    model = make_model()
    vectors = model.encode_items(batch_size=7)
    assert vectors.shape == (30, 8)
