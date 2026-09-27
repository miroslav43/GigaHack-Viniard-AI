import numpy as np
import torch

from fte.nn.augment import AugmentSpec, augment, dihedral
from fte.nn.losses import IGNORE, HeadLoss, head_loss, multi_head_loss


def test_head_loss_all_ignored_is_zero_and_finite():
    logits = torch.zeros(2, 8, 8, requires_grad=True)
    labels = torch.full((2, 8, 8), IGNORE, dtype=torch.uint8)
    loss = head_loss(logits, labels, HeadLoss())
    assert float(loss) == 0.0
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_head_loss_decreases_with_correct_logits():
    labels = torch.randint(0, 2, (1, 16, 16)).to(torch.uint8)
    good = (labels.float() * 2 - 1) * 6
    bad = -good
    spec = HeadLoss(focal_gamma=2.0)
    assert float(head_loss(good, labels, spec)) < float(head_loss(bad, labels, spec))


def test_multi_head_loss_shapes():
    logits = torch.zeros(1, 3, 4, 4)
    labels = torch.zeros(1, 3, 4, 4, dtype=torch.uint8)
    total, parts = multi_head_loss(logits, labels, [HeadLoss(), HeadLoss(0.5), HeadLoss(0.3)])
    assert len(parts) == 3 and torch.isfinite(total)


def test_dihedral_moves_labels_with_image():
    img = np.zeros((4, 4, 3), np.uint8)
    lbl = np.zeros((4, 4, 1), np.uint8)
    img[0, 3] = 255
    lbl[0, 3] = 1
    i2, l2 = dihedral(img, lbl, 1, True)
    assert (i2[..., 0] == 255).sum() == 1
    assert np.argwhere(i2[..., 0] == 255).tolist() == np.argwhere(l2[..., 0] == 1).tolist()


def test_augment_output_and_immutability():
    rng = np.random.default_rng(0)
    img = (rng.random((300, 300, 3)) * 255).astype(np.uint8)
    lbl = np.full((300, 300, 2), IGNORE, np.uint8)
    lbl[100:200, 100:200] = 1
    before = img.copy()
    out_img, out_lbl = augment(img, lbl, 256, AugmentSpec(), rng)
    assert out_img.shape == (256, 256, 3) and out_lbl.shape == (256, 256, 2)
    assert set(np.unique(out_lbl)) <= {1, IGNORE}
    assert np.array_equal(img, before)
