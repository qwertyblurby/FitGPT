import UploadIcon from 'assets/icons/UploadIcon';

function UploadButton({ onUpload }) {
	return (
		<form id="uploadForm" encType="multipart/form-data" className="size-full relative w-40 h-60">
			<label
				htmlFor="fileInput"
				className="btn rounded-lg dark:bg-gray-800 dark:text-gray-50 dark:hover:bg-gray-700 size-full flex flex-col"
			>
				<p className="text-gray-300">Upload</p>
				<UploadIcon className="w-16 h-16 text-gray-300" />
			</label>
			<input
				type="file"
				name="file"
				id="fileInput"
				accept="image/*"
				className="hidden"
				aria-label="Upload your image for outfit recommendations"
				onChange={onUpload}
			/>
		</form>
	)
}

export default UploadButton;
