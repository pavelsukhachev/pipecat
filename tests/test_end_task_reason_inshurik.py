#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Inshurik (LUCY-603): EndTaskReason keeps the members and values pipecat 1.3 had.

The voice app ends calls with these members, and its reports, callbacks and
dispositions match on the persisted strings, so the values are pinned here.
"""

import unittest

from pipecat.utils.enums import EndTaskReason


class TestEndTaskReasonInshurik(unittest.TestCase):
    def test_the_1_3_members_keep_their_values(self):
        self.assertEqual(EndTaskReason.END_CALL_TOOL_REASON.value, "end_call_tool")
        self.assertEqual(EndTaskReason.USER_QUALIFIED.value, "user_qualified")
        self.assertEqual(EndTaskReason.USER_DISQUALIFIED.value, "user_disqualified")
        self.assertEqual(EndTaskReason.SYSTEM_CONNECT_ERROR.value, "system_connect_error")
        self.assertEqual(EndTaskReason.UNKNOWN.value, "unknown")

    def test_upstream_members_are_untouched(self):
        self.assertEqual(EndTaskReason.END_CALL.value, "end_call")
        self.assertEqual(EndTaskReason.TRANSFER_CALL.value, "transfer_call")

    def test_values_stay_unique(self):
        values = [m.value for m in EndTaskReason]
        self.assertEqual(len(values), len(set(values)))


if __name__ == "__main__":
    unittest.main()
