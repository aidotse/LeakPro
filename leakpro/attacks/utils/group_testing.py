"""Group testing BCJR decoder wrapper."""

import ctypes
import os

import numpy as np
from numpy.ctypeslib import ndpointer


class GroupTestDecoder:
    """Handles per-sample BCJR decoding with dynamic parity matrices."""

    def __init__(self) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(current_dir, "group_test_decoder/BCJR_4_python.so")

        # Critical check: Verify file exists
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Library not found at: {lib_path}\n")
        # gcc -shared -o BCJR_4_python.so -fPIC BCJR_4_python.c

        # Check permissions (at least read access)
        if not os.access(lib_path, os.R_OK):
            raise PermissionError(
                    f"Missing read permissions for: {lib_path}\n")

        self.lib = ctypes.cdll.LoadLibrary(lib_path)
        self.configure_decoder()

    def configure_decoder(self) -> None:
        """Set up the BCJR function signature."""

        # Configure C function signature
        p_ui8_c = ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS")
        p_d_c = ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")
        self.lib.BCJR.argtypes = [
            p_ui8_c,  # H
            p_d_c,  # LLRinput
            p_ui8_c,   # test_values
            p_d_c,  # ChannelMatrix
            ctypes.c_double,             # threshold_dec
            ctypes.c_int,                # n_samples # n_clients
            ctypes.c_int,                # n_groups # n_tests
            p_d_c,  # LLRO
            p_ui8_c    # DEC
        ]

    def gt_decode(self,
                  H: np.ndarray,  # noqa: N803
                  binary_outcome: np.ndarray,
                  P_MD: float = 0.05,  # noqa: N803
                  P_FA: float = 0.05) -> float:  # noqa: N803
        """Decode one original sample's representatives."""
        # Get matrix dimensions
        n_tests, n_samples = H.shape

        # Initialize C buffers
        llr_out = np.zeros((1, n_samples), dtype=np.double)
        dec_out = np.zeros((1, n_samples), dtype=np.uint8)
        channel_matrix = np.array([[1-P_FA, P_FA], [P_MD, 1-P_MD]], dtype=np.double)

        # Validate test_values are group-level (n_tests,) not sample-level
        assert binary_outcome.shape == (n_tests,), \
            f"test_values must be size {n_tests}, got {binary_outcome.shape}"

        # Execute BCJR decoding
        self.lib.BCJR(
            H.astype(np.uint8, order="C"),
            np.log((1 - 0.1) / 0.1) * np.ones((1, n_samples), dtype=np.double),
            binary_outcome.astype(np.uint8, order="C"),
            channel_matrix,
            0.0,  # threshold_dec
            n_samples, # transformed samples,
            n_tests, # number of target model queries,
            llr_out,
            dec_out
        )

        # First flatten the array to make it 1D
        return llr_out.flatten()
