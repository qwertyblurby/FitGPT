import DemoData from './DemoData';

function DemoImage({ imageType, updateStatus }) {
	return (
		<button
			className="btn relative w-40 h-60 rounded-lg overflow-hidden p-0"
			onClick={(e) => {updateStatus(imageType)}}
		>
			<img
				src={DemoData(imageType).imageSrc}
				alt="Demonstration for outfit recommendations"
				className="object-cover w-full h-full"
			/>
		</button>
	)
}

export default DemoImage;
