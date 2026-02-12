import polars as pl
import os

SOURCE_FILE = "../data/train.parquet"
OUTPUT_FILE = "../data/train_undersampled_1_to_5.parquet"
RATIO = 5

print(f"1:{RATIO} 언더샘플링을 시작합니다...")
print(f"원본 데이터: {SOURCE_FILE}")
print(f"저장 경로: {OUTPUT_FILE}")
print("-" * 60)

try:
    # Lazy frame으로 파일 스캔
    lf = pl.scan_parquet(SOURCE_FILE)

    # clicked = 1인 행의 개수 파악
    print("clicked = 1인 행의 개수를 계산하는 중...")
    n_ones = lf.filter(pl.col("clicked") == 1).select(pl.len()).collect().item()

    if n_ones == 0:
        print("clicked = 1인 행을 찾을 수 없습니다.")
    else:
        n_zeros_to_sample = n_ones * RATIO
        print(f"clicked = 1: {n_ones:,}개")
        print(f"clicked = 0 샘플링: {n_zeros_to_sample:,}개")
        print(f"최종 데이터셋 크기: {n_ones + n_zeros_to_sample:,}개")
        print(f"예상 클릭률: {1/(1+RATIO):.1%}")

        print("\n 데이터 샘플링 중...")

        # clicked = 1인 모든 데이터
        lazy_ones = lf.filter(pl.col("clicked") == 1)

        # clicked = 0인 데이터에서 랜덤 샘플링
        lazy_zeros_sampled = (
            lf.filter(pl.col("clicked") == 0)
            .with_row_index("index")
            .with_columns(
                pl.col("index").shuffle(seed=42).alias("random_order")
            )
            .sort("random_order")
            .head(n_zeros_to_sample)
            .drop(["index", "random_order"])
        )

        print("데이터 결합 및 최종 셔플링 중...")

        # 두 데이터셋 결합 후 최종 셔플링을 위해 collect 후 다시 lazy로
        combined_df = pl.concat([
            lazy_ones,
            lazy_zeros_sampled
        ]).collect()

        # 셔플링 후 다시 lazy frame으로
        final_undersampled_lazy = combined_df.sample(fraction=1.0, shuffle=True, seed=42).lazy()

        print("parquet 파일로 저장 중...")

        # 파케트 파일로 저장
        final_undersampled_lazy.sink_parquet(OUTPUT_FILE)

        # 저장된 파일 검증
        print("\n 저장된 파일 검증 중...")
        verification = pl.scan_parquet(OUTPUT_FILE).select([
            pl.len().alias("total_rows"),
            pl.col("clicked").sum().alias("clicked_1_count"),
            pl.col("clicked").mean().alias("click_rate"),
            (pl.col("clicked") == 0).sum().alias("clicked_0_count")
        ]).collect()

        total_rows = verification["total_rows"][0]
        clicked_1_count = verification["clicked_1_count"][0]
        clicked_0_count = verification["clicked_0_count"][0]
        click_rate = verification["click_rate"][0]

        print("=" * 60)
        print("언더샘플링 완료!")
        print(f"저장 경로: {os.path.abspath(OUTPUT_FILE)}")
        print(f"최종 데이터셋 정보:")
        print(f"   • 총 행 수: {total_rows:,}개")
        print(f"   • clicked = 1: {clicked_1_count:,}개 ({click_rate:.2%})")
        print(f"   • clicked = 0: {clicked_0_count:,}개 ({(1-click_rate):.2%})")
        print(f"   • 클래스 비율 (0:1): {clicked_0_count/clicked_1_count:.1f}:1")
        print(f"   • 목표 비율 달성: {'✅' if abs(clicked_0_count/clicked_1_count - RATIO) < 0.1 else '❌'}")
        print("=" * 60)

except Exception as e:
    print(f"오류 발생: {e}")
    print("파일 경로나 권한을 확인해주세요.")
