import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Adicionar o diretório raiz ao path para importar os módulos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eye_mouse')))

from config import get_resource_path, get_user_data_dir

class TestConfig:
    def test_get_resource_path_dev(self):
        """Teste get_resource_path em ambiente de desenvolvimento (sem PyInstaller)"""
        # Garantir que _MEIPASS não existe
        if hasattr(sys, '_MEIPASS'):
            del sys._MEIPASS
            
        path = get_resource_path("test_file.txt")
        expected = os.path.join(os.path.abspath("."), "test_file.txt")
        assert path == expected

    def test_get_resource_path_pyinstaller(self):
        """Teste get_resource_path em ambiente PyInstaller (com _MEIPASS)"""
        # Simular _MEIPASS
        sys._MEIPASS = "/tmp/meipass"
        
        path = get_resource_path("test_file.txt")
        expected = os.path.join("/tmp/meipass", "test_file.txt")
        assert path == expected
        
        # Limpar
        del sys._MEIPASS

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('os.path.expanduser')
    def test_get_user_data_dir_creates_dir(self, mock_expanduser, mock_exists, mock_makedirs):
        """Teste se get_user_data_dir cria o diretório se não existir"""
        mock_expanduser.return_value = "/home/user"
        mock_exists.return_value = False
        
        path = get_user_data_dir()
        
        expected_path = os.path.join("/home/user", "Documents", "EyeMouse")
        assert path == expected_path
        mock_makedirs.assert_called_once_with(expected_path)

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('os.path.expanduser')
    def test_get_user_data_dir_exists(self, mock_expanduser, mock_exists, mock_makedirs):
        """Teste se get_user_data_dir não cria diretório se já existir"""
        mock_expanduser.return_value = "/home/user"
        mock_exists.return_value = True
        
        path = get_user_data_dir()
        
        expected_path = os.path.join("/home/user", "Documents", "EyeMouse")
        assert path == expected_path
        mock_makedirs.assert_not_called()
