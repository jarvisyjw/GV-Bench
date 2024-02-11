python -m hloc.pairs_from_retrieval \
                  --descriptors dataset/robotcar/features/qAutumn_dbRain/netvlad.h5 \
                  --query_prefix Autumn_mini_val \
                  --db_prefix Rain_mini_val \
                  --output dataset/robotcar/pairs/rain_test.txt \
                  --num_matched 20