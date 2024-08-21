import PyPDF2

def remove_pages(input_pdf_path, output_pdf_path, pages_save):
    """
    Removes specified pages from a PDF file and saves the rest to a new PDF file.

    :param input_pdf_path: Path to the input PDF file.
    :param output_pdf_path: Path to the output PDF file.
    :param pages_to_remove: A list of page numbers (0-indexed) to remove.
    """
    with open(input_pdf_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        writer = PyPDF2.PdfWriter()

        # Iterate over all pages and add to writer if not in pages_to_remove
        for page_num in range(len(reader.pages)):
            if page_num in pages_save:
                writer.add_page(reader.pages[page_num])

        # Write the output PDF file
        with open(output_pdf_path, 'wb') as output_file:
            writer.write(output_file)

# Example usage
input_pdf = r'D:\tmp\Buffer of Thoughts.pdf'
output_pdf = r'D:\tmp\Buffer of Thoughts01.pdf'
pages_to_remove = [0,1]  # Remove pages 0, 2, and 4 (0-indexed)
remove_pages(input_pdf, output_pdf, pages_to_remove)