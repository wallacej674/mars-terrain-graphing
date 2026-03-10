import unittest
from unittest.mock import MagicMock, patch

import download_tiles


class _Obj:
    def __init__(self, key):
        self.key = key


class TestDownloadTiles(unittest.TestCase):
    @patch("download_tiles.boto3.client")
    @patch("download_tiles.boto3.resource")
    @patch("download_tiles.os.makedirs")
    def test_download_filters_and_downloads_expected_keys(self, mock_makedirs, mock_resource, mock_client):
        keys = [
            "mars/mro/ctx/controlled/usgs/a/a-DEM.tif",
            "mars/mro/ctx/controlled/usgs/a/a-DRG.tif",
            "mars/mro/ctx/controlled/usgs/a/GoodPixelMap.tif",
            "mars/mro/ctx/controlled/usgs/a/not_dem.tif",
            "mars/mro/ctx/controlled/usgs/a/readme.xml",
            "mars/mro/ctx/controlled/usgs/sub/b-DEM.TIF",
        ]
        objects = [_Obj(k) for k in keys]

        mock_bucket = MagicMock()
        mock_bucket.objects.filter.return_value = objects
        mock_resource.return_value.Bucket.return_value = mock_bucket
        mock_s3_client = MagicMock()
        mock_client.return_value = mock_s3_client

        download_tiles.download_ctx_dtms()

        # Root data directory creation + nested directory creation for valid downloads.
        self.assertTrue(mock_makedirs.called)

        expected_calls = [
            (
                download_tiles.BUCKET_NAME,
                "mars/mro/ctx/controlled/usgs/a/a-DEM.tif",
                "data\\a/a-DEM.tif",
            ),
            (
                download_tiles.BUCKET_NAME,
                "mars/mro/ctx/controlled/usgs/sub/b-DEM.TIF",
                "data\\sub/b-DEM.TIF",
            ),
        ]
        actual = [c.args for c in mock_s3_client.download_file.call_args_list]
        self.assertEqual(actual, expected_calls)
        self.assertEqual(mock_s3_client.download_file.call_count, 2)


if __name__ == "__main__":
    unittest.main()
