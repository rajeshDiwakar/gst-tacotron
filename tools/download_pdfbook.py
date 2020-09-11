import os,sys
from googlesearch import search
import csv
import time

def download_pdfbooks(file_path):



    log_file = os.path.splitext(file_path)[0]+'_pdflog.tsv'
    if os.path.isfile(log_file):
        mv_path,ext = os.path.splitext(log_file)
        mv_path = mv_path+ '%s_bk'%time.asctime() + ext
        print('Moving %s to %s'%(log_file,mv_path))
        os.rename(log_file,mv_path)
    with open(log_file,'w',encoding='utf-8') as f_log:
        with open(file_path) as csvfile:
            dialect = csv.Sniffer().sniff(csvfile.read(1024))
            csvfile.seek(0)
            reader = csv.reader(csvfile, dialect)
            csvwriter = csv.writer(f_log,dialect)

            for row in reader:
                print(row)
                if len(row)<3:
                    csvwriter.writerow(row+[''])
                    print('unexpected number of cols\n%s'%str(row))
                    continue
                i,book,author = row
                query = '%s %s ext:pdf'%(book,author)
                try:
                    urls = [url for url in search(query, tld="co.in", num=10, stop=10, pause=2) if url.lower().endswith('.pdf')]
                    csvwriter.writerow(row + [' '.join(urls[:3])])
                except Exception as e:
                    print(str(e))


if __name__=='__main__':
    # file_path = '/home/rajesh/work/limbo/data/yt/500books.tsv'
    # file_path = '/home/rajesh/work/limbo/data/yt/testbooks2.tsv'
    try:
        file_path = sys.argv[1]
    except Exception as e:
        print('python %s path/to/book.tsv'%(sys.argv[0]))
    download_pdfbooks(file_path)
