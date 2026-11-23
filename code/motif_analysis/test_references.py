"""Offline reference, biological-identifier and full-workflow coverage tests."""
import csv
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import references as r


class ReferenceTests(unittest.TestCase):
    def write_meme(self, path, identifier='MA0001.1'):
        path.write_text('MEME version 4\nALPHABET= ACGT\nMOTIF '+identifier+' Test\n'
                        'letter-probability matrix: alength= 4 w= 2\n'
                        '.25 .25 .25 .25\n1 0 0 0\n')

    def test_valid_probability_matrix_and_version(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'motifs.txt'; self.write_meme(p)
            self.assertEqual(r.meme_profiles(p)['MA0001.1']['width'],2)

    def test_html_download_is_not_a_motif_database(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'motifs.txt';p.write_text('<html>Not Found</html>')
            with self.assertRaises(RuntimeError):r.meme_profiles(p)

    def test_truncated_matrix_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'motifs.txt';self.write_meme(p)
            p.write_text(p.read_text().replace('w= 2','w= 3'))
            with self.assertRaises(RuntimeError):r.meme_profiles(p)

    def test_exact_jaspar_id_and_species_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'db.sqlite3';db=sqlite3.connect(p)
            db.executescript('''CREATE TABLE MATRIX(ID INTEGER,BASE_ID TEXT,VERSION INTEGER,COLLECTION TEXT,NAME TEXT);
                CREATE TABLE MATRIX_PROTEIN(ID INTEGER,ACC TEXT);
                CREATE TABLE MATRIX_SPECIES(ID INTEGER,TAX_ID INTEGER);
                CREATE TABLE MATRIX_ANNOTATION(ID INTEGER,TAG TEXT,VAL TEXT);
                INSERT INTO MATRIX VALUES(1,'MA0001',1,'CORE','TF');
                INSERT INTO MATRIX VALUES(2,'MA0001',2,'CORE','TF_UPDATED');
                INSERT INTO MATRIX_PROTEIN VALUES(1,'P12345');
                INSERT INTO MATRIX_SPECIES VALUES(1,10090);
                INSERT INTO MATRIX_ANNOTATION VALUES(1,'tax_group','vertebrates');''')
            db.commit();db.close()
            found=r.sql_metadata(p,{'MA0001.1':{}})['MA0001.1']
            self.assertEqual(found['name'],'TF')
            self.assertEqual(found['proteins'],{'P12345'})
            self.assertEqual(found['species'],{'10090'})
            with self.assertRaises(RuntimeError):r.sql_metadata(p,{'MA0001.99':{}})

    def test_species_restriction_in_motif_database(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'db.sqlite3';db=sqlite3.connect(p)
            db.executescript('''CREATE TABLE MATRIX(ID,BASE_ID,VERSION,COLLECTION,NAME);
                CREATE TABLE MATRIX_PROTEIN(ID,ACC); CREATE TABLE MATRIX_SPECIES(ID,TAX_ID);
                CREATE TABLE MATRIX_ANNOTATION(ID,TAG,VAL);
                INSERT INTO MATRIX VALUES(1,'MA0001',1,'CORE','TF');
                INSERT INTO MATRIX_ANNOTATION VALUES(1,'tax_group','plants');''')
            db.commit();db.close()
            with self.assertRaises(RuntimeError):r.sql_metadata(p,{'MA0001.1':{}})

    def test_protein_based_mouse_candidates_preserve_one_to_many_homology(self):
        homology=({'P12345':{('Genea','MGI_human_UniProt_to_mouse_homology','7'),
                              ('Geneb','MGI_human_UniProt_to_mouse_homology','7')}},{},{})
        mapping,audit=r.build_mapping({'MA0001.1':{'name':'UNFAMILIAR'}},
            {'MA0001.1':dict(name='UNFAMILIAR',proteins={'P12345'},species={'9606'})},homology)
        self.assertEqual(mapping[0]['Mouse_JASPAR_MGI_candidates'],'Genea, Geneb')
        self.assertEqual(len(audit),2)

    def test_other_species_name_is_not_guessed_as_a_mouse_tf(self):
        homology=({}, {'known':{'Known'}}, {'known':{'Known'}})
        mapping,audit=r.build_mapping({'MA0001.1':{'name':'KNOWN'}},
            {'MA0001.1':dict(name='KNOWN',proteins=set(),species={'7955'})},homology)
        self.assertEqual(mapping[0]['status'],'no_verified_mouse_assignment')
        self.assertFalse(audit)

    def test_source_mouse_symbol_requires_unique_approved_mapping(self):
        homology=({}, {'known':{'Known','Known2'}}, {})
        mapping,_=r.build_mapping({'MA0001.1':{}},
            {'MA0001.1':dict(name='KNOWN',proteins=set(),species={'10090'})},homology)
        self.assertEqual(mapping[0]['Mouse_JASPAR_MGI_candidates'],'')

    def test_mgi_report_schema_and_protein_mapping(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'homology.rpt'
            with p.open('w',newline='') as f:
                w=csv.writer(f,delimiter='\t')
                w.writerow(['DB Class Key','NCBI Taxon ID','Symbol','SWISS-PROT IDs'])
                for i in range(1002):
                    w.writerow([i,10090,'Mouse'+str(i),'Q'+str(i).zfill(5)])
                w.writerow([0,9606,'HUMAN','P99999'])
            proteins,mice,human=r.mouse_homology(p)
            self.assertIn(('Mouse0','MGI_human_UniProt_to_mouse_homology','0'),proteins['P99999'])
            self.assertEqual(human['human'],{'Mouse0'})

    def test_download_failure_does_not_keep_partial_or_create_manifest(self):
        with tempfile.TemporaryDirectory() as d,patch.object(r.urllib.request,'urlopen',side_effect=OSError('offline')):
            with self.assertRaises(RuntimeError):r.fetch(Path(d))
            self.assertFalse(list(Path(d).iterdir()))

    def test_reference_hash_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);f=p/'file';f.write_text('original')
            r.dump(p/'REFERENCE_MANIFEST.json',dict(release=2026,profiles=1019,files={'file':r.sha(f)}))
            r.verify(p);f.write_text('changed')
            with self.assertRaises(RuntimeError):r.verify(p)

    def test_references_precede_any_submission(self):
        text=(r.P/'SUBMIT_BETTY.sh').read_text()
        self.assertLess(text.index('references.py'),text.index('control.py'))
        self.assertIn('set -euo pipefail',text)


if __name__=='__main__':unittest.main()
