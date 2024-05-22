function Profile({ name, bio, imageSrc }) {
	return (
		<div className="flex flex-row items-center mt-6 gap-4">
			<img
				alt={name}
				className="aspect-square rounded-full object-cover"
				height="64"
				width="64"
				src={imageSrc.includes("http") ? imageSrc : require(`./assets/about/${imageSrc}`)}
			/>
			<div>
				<h3 className="text-lg font-semibold">{name}</h3>
				<p className="text-gray-400">{bio}</p>
			</div>
		</div>
	);
}

export default Profile;
