# 打开原始文件和新文件
with open('../data/CAMI/oral/oral6/anonymous_gsa-2mm.fasta', 'r') as f_in, open('../data/CAMI/oral/oral6/anonymous_gsa-2mm-new.fasta', 'w') as f_out:
    # 逐行读取原始文件内容
    for line in f_in:
        # 去除每行末尾的换行符并写入新文件
        f_out.write(line.rstrip('\n') + '\r')
