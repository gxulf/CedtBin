# 定义输入和输出文件路径
input_file = "../data/CAMI/oral/oral6/anonymous_gsa-2mm.fasta"
output_file = "../data/CAMI/oral/oral6/anonymous_gsa-16mm.fasta"

# 读取输入文件的内容
with open(input_file, "r") as f:
    lines = f.readlines()

# 将标题行和序列数据行组合为一个整体
contigs = []
for i in range(0, len(lines), 2):
    # 检查是否存在足够的行来构成一个 contig
    if i + 1 < len(lines):
        contig_name = lines[i].strip()
        sequence = lines[i + 1].strip()
        contigs.append((contig_name, sequence))
    else:
        print("警告：文件格式不正确，可能存在未完整的 contig")

# 计算要保留的 contigs 数量（保留一半）1/4
keep_contigs = len(contigs) // 8

# 截取一半的 contigs 数据
contigs = contigs[:keep_contigs]


# 将截取后的 contigs 写入输出文件
with open(output_file, "w") as f:
    for contig in contigs:
        f.write(contig[0] + '\n')
        f.write(contig[1] + '\n')