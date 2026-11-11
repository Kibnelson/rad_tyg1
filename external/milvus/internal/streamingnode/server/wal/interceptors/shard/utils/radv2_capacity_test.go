package utils

import (
	"math"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
)

func TestRADV2PhysicalRowBoundary(t *testing.T) {
	s := &SegmentStats{Level: datapb.SegmentLevel_L1, MaxRows: 1000000, MaxBinarySize: math.MaxUint64}
	require.True(t, s.AllocRows(ModifiedMetrics{Rows: 999990, BinarySize: 100}))
	require.False(t, s.ShouldBeSealed())
	// A whole crossing request must not prematurely seal or change occupancy.
	require.False(t, s.AllocRows(ModifiedMetrics{Rows: 11, BinarySize: 10}))
	require.Equal(t, uint64(999990), s.Modified.Rows)
	require.False(t, s.ShouldBeSealed())
	require.True(t, s.AllocRows(ModifiedMetrics{Rows: 10, BinarySize: 10}))
	require.True(t, s.ShouldBeSealed())
	require.Equal(t, uint64(1000000), s.Modified.Rows)
	require.False(t, s.AllocRows(ModifiedMetrics{Rows: 1, BinarySize: 1}))
	// Recovery preserves both the physical row bound and the full-segment seal.
	restored := NewSegmentStatFromProto(NewProtoFromSegmentStat(s))
	require.True(t, restored.HasL1RowLimit())
	require.True(t, restored.ShouldBeSealed())
	require.Equal(t, uint64(1000000), restored.MaxRows)
	require.False(t, restored.AllocRows(ModifiedMetrics{Rows: 1, BinarySize: 1}))
}

func TestRADV2PreservesUpstreamByteAndL0Policies(t *testing.T) {
	legacyRows := &SegmentStats{Level: datapb.SegmentLevel_L1, MaxRows: 10, MaxBinarySize: 100}
	require.False(t, legacyRows.HasL1RowLimit())
	byteMode := &SegmentStats{Level: datapb.SegmentLevel_L1, MaxRows: math.MaxUint64, MaxBinarySize: 10}
	require.False(t, byteMode.HasL1RowLimit())
	require.True(t, byteMode.AllocRows(ModifiedMetrics{Rows: 1, BinarySize: 10}))
	require.False(t, byteMode.ShouldBeSealed())
	require.False(t, byteMode.AllocRows(ModifiedMetrics{Rows: 1, BinarySize: 1}))
	require.True(t, byteMode.ShouldBeSealed())
	l0 := &SegmentStats{Level: datapb.SegmentLevel_L0, MaxRows: 10, MaxBinarySize: 100}
	require.False(t, l0.HasL1RowLimit())
	require.True(t, l0.AllocRows(ModifiedMetrics{Rows: 10, BinarySize: 10}))
	require.False(t, l0.ShouldBeSealed())
	require.False(t, l0.AllocRows(ModifiedMetrics{Rows: 1, BinarySize: 1}))
	require.True(t, l0.ShouldBeSealed())
}
