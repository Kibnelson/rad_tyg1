package shards

import (
	"math"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

func TestRADV2L1RowCapacityPolicy(t *testing.T) {
	paramtable.Init()
	params := paramtable.Get()
	key := params.StreamingCfg.FlushL1MaxRowNum.Key
	t.Cleanup(func() { params.Reset(key) })
	params.Save(key, "1000000")
	// Byte thresholds, seal proportion and random size jitter cannot change rows.
	p := jitterSegmentLimitationPolicy{}.GenerateLimitation(datapb.SegmentLevel_L1)
	require.Equal(t, "radv2_fixed_l1_rows", p.PolicyName)
	require.Equal(t, uint64(1000000), p.SegmentRows)
	require.Equal(t, uint64(math.MaxUint64), p.SegmentSize)
	params.Save(key, "0")
	legacy := jitterSegmentLimitationPolicy{}.GenerateLimitation(datapb.SegmentLevel_L1)
	require.Equal(t, uint64(math.MaxUint64), legacy.SegmentRows)
	require.Less(t, legacy.SegmentSize, uint64(math.MaxUint64))
	// The new L1 parameter has no effect on L0 delete-segment sizing.
	params.Save(key, "1000000")
	l0 := jitterSegmentLimitationPolicy{}.GenerateLimitation(datapb.SegmentLevel_L0)
	require.Equal(t, "jitter_segment_limitation", l0.PolicyName)
}
