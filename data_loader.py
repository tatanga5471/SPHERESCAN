from astropy.io import fits
from astropy.utils.data import download_file
from astropy.wcs import WCS
import os
import io
from utils import logger


def get_sample_fits_path():
    try:
        url = 'http://data.astropy.org/tutorials/FITS-images/HorseHead.fits'
        file_path = download_file(url, cache=True)
        return file_path
    except Exception as e:
        logger.error(f"Örnek FITS indirme hatası: {e}")
        return None


def load_fits_from_file(uploaded_file):
    try:
        # Streamlit yükleme nesnesini güvenli byte akışına dönüştür
        bytes_data = uploaded_file.read()
        with fits.open(io.BytesIO(bytes_data)) as hdul:
            # Multi-Extension FITS kontrolü: Veri içeren ilk geçerli HDU'yu bul
            target_hdu = None
            for hdu in hdul:
                if hdu.data is not None and len(hdu.data.shape) >= 2:
                    target_hdu = hdu
                    break

            if target_hdu is None:
                target_hdu = hdul[0]

            header = target_hdu.header
            data = target_hdu.data

            # Eğer 3D spektral küpse (örn. SPHEREx), ilk katmanı (index 0) 2D görüntü olarak al
            if data is not None and len(data.shape) == 3:
                logger.warning(f"3D Spektral Küp algılandı (Boyut: {data.shape}). İlk bant seçiliyor.")
                data = data[0]
            elif data is not None and len(data.shape) > 3:
                data = data[tuple(0 for _ in range(data.ndim - 2))]

            return header, data
    except Exception as e:
        logger.error(f"FITS dosya okuma hatası: {e}")
        return None, None


def read_fits_data(file_path):
    if not os.path.exists(file_path):
        logger.error(f"Dosya bulunamadı: {file_path}")
        return None, None
    try:
        with fits.open(file_path) as hdul:
            target_hdu = None
            for hdu in hdul:
                if hdu.data is not None and len(hdu.data.shape) >= 2:
                    target_hdu = hdu
                    break
            if target_hdu is None:
                target_hdu = hdul[0]

            header = target_hdu.header
            data = target_hdu.data

            if data is not None and len(data.shape) == 3:
                data = data[0]

            return header, data
    except Exception as e:
        logger.error(f"FITS açma hatası: {e}")
        return None, None


def get_wcs_coordinates(header, x, y):
    try:
        wcs = WCS(header)
        sky_coord = wcs.pixel_to_world(x, y)
        return float(sky_coord.ra.deg), float(sky_coord.dec.deg)
    except Exception:
        return None, None