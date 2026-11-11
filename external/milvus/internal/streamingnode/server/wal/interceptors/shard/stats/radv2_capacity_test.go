package stats

import (
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/internal/mocks/streamingnode/server/wal/interceptors/shard/mock_utils"
	"github.com/milvus-io/milvus/internal/streamingnode/server/wal/interceptors/shard/policy"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/streaming/util/types"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

func TestRADV2AllocationDoesNotRollUnderfilledSegment(t *testing.T) {
	paramtable.Init()
	m := NewStatsManager()
	op := mock_utils.NewMockSealOperator(t)
	op.EXPECT().Channel().Return(types.PChannelInfo{Name: "radv2"}).Maybe()
	op.EXPECT().AsyncFlushSegment(mock.Anything).Return().Maybe()
	m.RegisterSealOperator(op, nil, nil)
	t.Cleanup(func() { m.UnregisterSealOperator(op) })
	s := &SegmentStats{Level: datapb.SegmentLevel_L1, MaxRows: 1000000, MaxBinarySize: math.MaxUint64,
		Modified:   ModifiedMetrics{Rows: 999990, BinarySize: 100},
		CreateTime: time.Now(), LastModifiedTime: time.Now()}
	m.RegisterNewGrowingSegment(SegmentBelongs{PChannel: "radv2", VChannel: "v", CollectionID: 1, PartitionID: 1, SegmentID: 1}, s)
	sealed, err := m.allocRows(1, ModifiedMetrics{Rows: 11, BinarySize: 11}, 11)
	require.ErrorIs(t, err, ErrTooLargeInsert)
	require.False(t, sealed)
	require.Equal(t, uint64(999990), m.GetStatsOfSegment(1).Modified.Rows)
	sealed, err = m.allocRows(1, ModifiedMetrics{Rows: 10, BinarySize: 10}, 10)
	require.NoError(t, err)
	require.True(t, sealed)
	require.Equal(t, uint64(1000000), m.GetStatsOfSegment(1).Modified.Rows)
}

func TestRADV2RowModeIgnoresIdleAndLifetimeSealing(t *testing.T) {
	paramtable.Init()
	m := NewStatsManager()
	old := time.Now().Add(-100 * 24 * time.Hour)
	m.segmentStats[1] = &SegmentStats{Level: datapb.SegmentLevel_L1, MaxRows: 1000000, MaxBinarySize: math.MaxUint64,
		Modified: ModifiedMetrics{Rows: 17, BinarySize: 1 << 30}, CreateTime: old, LastModifiedTime: old}
	m.segmentStats[2] = &SegmentStats{Level: datapb.SegmentLevel_L1, MaxRows: 1000000, MaxBinarySize: math.MaxUint64,
		Modified: ModifiedMetrics{Rows: 1000000}, CreateTime: old, LastModifiedTime: old}
	m.segmentStats[3] = &SegmentStats{Level: datapb.SegmentLevel_L0, MaxRows: 1000000,
		Modified: ModifiedMetrics{Rows: 17}, CreateTime: old, LastModifiedTime: old}
	selected := m.selectSegmentsWithTimePolicy()
	require.NotContains(t, selected, int64(1))
	require.Equal(t, policy.PolicyNameCapacity, selected[2].Policy)
	require.Equal(t, policy.PolicyNameLifetime, selected[3].Policy)
	// No resource or explicit-flush policy was removed; only normal time/binlog
	// sealing is excluded from fixed-row mode.
}
