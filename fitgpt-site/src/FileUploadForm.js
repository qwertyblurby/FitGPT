function FileUploadForm({ onUpload }) {
	return (
		<form id="uploadForm" encType="multipart/form-data" className="flex items-center gap-4">
			{/* File input field */}
			
			<input
				type="file"
				name="file"
				id="fileInput"
				accept="image/*"
				className="hidden"
				aria-label="Upload Your Image"
				onChange={onUpload}
				
			/>
			{/* Daisy UI button style */}
			<label htmlFor="fileInput" className="btn dark:bg-gray-800 dark:text-gray-50 dark:hover:bg-gray-700">
				Upload Your Image
			</label>
			
			
			
			{/* Message for file uploaded */}
			{/* <p id="fileUploadedMessage" style={{ display: 'none', color: 'green' }}>File uploaded successfully!</p> */}
		</form>
	)
}

export default FileUploadForm;
