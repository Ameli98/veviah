"""VrmRenderer against the real headless player. Skipped when Node or the player's packages are missing."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_loop import test_motions as tm
from motion_loop.render import render_grid
from motion_loop.vrm_render import PLAYER_DIR, TALL_CROP, VrmRenderer, find_node

READY = bool(find_node()) and (PLAYER_DIR / "node_modules" / "playwright").exists()


@unittest.skipUnless(READY, "needs node and `npm install` in vrm_player/")
class VrmRender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = VrmRenderer(size=192)

    @classmethod
    def tearDownClass(cls):
        cls.r.close()

    def test_frame_is_front_side_and_close_up(self):
        f = self.r.render_frames(tm.nod(), [0])[0]
        self.assertEqual(f.size, (576, 192))

    def test_raised_hand_is_on_screen_left_of_front_view(self):
        """Character faces the camera, so its right hand must appear on the left. Catches mirroring."""
        m = tm.raise_right_hand()
        rest, up = (np.asarray(f, float)[:, :192] for f in self.r.render_frames(m, [0, len(m) // 2]))
        changed = np.abs(up - rest).sum(-1) > 60
        top = changed[:96]                                   # above the shoulders
        left, right = int(top[:, :96].sum()), int(top[:, 96:].sum())
        self.assertGreater(left, 80, (left, right))          # the thin raised arm, in a 192 px view
        self.assertGreater(left, 10 * max(right, 1), (left, right))

    def test_same_input_same_image(self):
        a, b = (self.r.render_frames(tm.wave(), [10, 50]) for _ in range(2))
        for x, y in zip(a, b):
            self.assertTrue(np.array_equal(np.asarray(x), np.asarray(y)))

    def test_grid_layout(self):
        g = render_grid(self.r, tm.wave(), 8)
        self.assertEqual(g.size, (576 * 2, 192 * 4))

    def test_tall_crop_is_the_cut_square_view(self):
        """--tall videos: rendering the crop must give the same picture as cutting the square render.
        Size 200 so that every crop edge falls on a whole pixel (200 * 0.125, 0.75, 0.9)."""
        sq, tall = VrmRenderer(size=200), VrmRenderer(size=200, crop=TALL_CROP)
        try:
            s = np.asarray(sq.render_frames(tm.wave(), [30])[0], int)
            t = np.asarray(tall.render_frames(tm.wave(), [30])[0], int)
        finally:
            sq.close(); tall.close()
        self.assertEqual(t.shape, (180, 3 * 150, 3))
        cut = np.concatenate([s[:180, v * 200 + 25:v * 200 + 175] for v in range(3)], axis=1)
        diff = np.abs(cut - t).max(-1)
        self.assertLess((diff > 40).mean(), 0.002)          # only antialiased edge pixels may differ

if __name__ == "__main__":
    unittest.main()
