import pandas as pd
from pathlib import Path
import math
from .utils import parse_pairs_from_retrieval
from . import logger
from tqdm import tqdm


def utm(df: pd.DataFrame, key: str) -> pd.Series:
    return df[df['key'] == key]['easting'].item(), df[df['key'] == key]['northing'].item()

def cdist(easting1, northing1, easting2, northing2):
    return math.sqrt((easting1 - easting2)**2 + (northing1 - northing2)**2)

def name2key(name: str):
    return name.split('/')[-1].split('.')[0]

def main(pairs_path, query_df, database_df, fout):
      pairs_loader = parse_pairs_from_retrieval(pairs_path)
      for query, database in tqdm(pairs_loader):
            logger.debug('Processing pair: %s, %s', query, database)
            logger.debug('Query: %s', name2key(query))
            logger.debug('Database: %s', name2key(database))
            q_easting, q_northing = utm(query_df, name2key(query))
            db_easting, db_northing = utm(database_df, name2key(database))
            logger.debug(f'Query: {q_easting}, {q_northing}')
            logger.debug(f'Database: {db_easting}, {db_northing}')
            distance = cdist(q_easting, q_northing, db_easting, db_northing)
            logger.debug(f'Distacne: {distance}')
            if distance < 25:
                  fout.write(f'{query} {database} 1\n')
            else:
                  fout.write(f'{query} {database} 0\n')
      fout.close()
      logger.info('Done')

if __name__ == "__main__":
      city = 'boston'
      root_path = Path('dataset/mapillary_sls/train_val')
      pairs_path = 'dataset/mapillary_sls/pairs/boston.txt'
      
      query_df = pd.read_csv(root_path / city / 'query' / 'postprocessed.csv', index_col = 0)
      database_df = pd.read_csv(root_path / city / 'database' / 'postprocessed.csv', index_col = 0)
      # database_df = pd.read_csv('dataset/mapillary_sls/train_val/boston/database/postprocessed.csv', index_col = 0)

      fout = open('output.txt', 'w')
      main(pairs_path, query_df, database_df, fout)


# csv_path = 'dataset/mapillary_sls/train_val/boston/query/postprocessed.csv'
# a = pd.read_csv(csv_path, index_col = 0)
# print(a)
# print(a[2], a[1])
# qData = pd.read_csv(join(root_dir, subdir, city, 'query', 'postprocessed.csv'), index_col = 0)
# output_list = [i for i in a]
# print(a)
# print(a[a['key'] == 'Bd7m8vNyPIJaPBXLw5ghnw'])