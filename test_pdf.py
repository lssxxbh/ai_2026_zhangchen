import fitz  # PyMuPDF
from pathlib import Path

print("="*60)
print("PDF文本提取测试工具")
print("="*60)

# 检查uploads目录
uploads_dir = Path(__file__).parent / "uploads"
print(f"\n检查上传目录: {uploads_dir}")

if not uploads_dir.exists():
    print("上传目录不存在")
else:
    files = list(uploads_dir.glob("*.pdf"))
    print(f"找到 {len(files)} 个PDF文件")
    for f in files:
        print(f"  - {f.name}")

print("\n" + "="*60)

# 如果有PDF文件，测试第一个
if len(files) > 0:
    test_pdf = files[-1]  # 最新的一个
    print(f"\n测试文件: {test_pdf}")
    print(f"文件大小: {test_pdf.stat().st_size} bytes")
    
    try:
        doc = fitz.open(test_pdf)
        print(f"\nPDF信息:")
        print(f"  页数: {len(doc)}")
        print(f"  元数据: {doc.metadata}")
        
        all_text = []
        for i, page in enumerate(doc):
            print(f"\n第 {i+1} 页:")
            text = page.get_text()
            print(f"  文本长度: {len(text)}")
            print(f"  文本内容:\n{repr(text[:500])}")
            if text.strip():
                all_text.append(text)
        
        full_text = "\n".join(all_text)
        print(f"\n总文本长度: {len(full_text)}")
        print(f"总文本前1000字符:\n{full_text[:1000]}")
        
        doc.close()
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
